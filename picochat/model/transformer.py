"""The assembled picochat model: TransformerLayer (one mixer + FFN/MoE
sublayer pair under Block Attention Residuals), Transformer (the interleaved
3-GDN2 : 1-NSA hybrid stack), the multi-token-prediction heads and
TransformerLM. Building blocks live in blocks.py / moe.py, the mixers in
linear_attn.py / sparse_attn.py; the scale-ladder presets and the build_lm
factory in presets.py, the parameter estimator in estimate.py. The
LightningModules that train it live in picochat.training.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from picochat.model.blocks import (
    DepthAttention,
    SwiGLU,
    doc_ids_to_cu_seqlens,
    rms_norm,
)
from picochat.model.linear_attn import GatedDeltaNet2
from picochat.model.moe import ExpertBank, MixtureOfExperts
from picochat.model.sparse_attn import NativeSparseAttention


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        linear: bool,
        n_kv_heads: int | None = None,
        nsa_kv_heads: int = 1,
        d_ffn: int | None = None,
        max_seq_len: int = 4096,
        window_size: int = 512,
        rope_base: float = 1_000_000.0,
        rope_factor: float = 0.25,
        sel_block: int = 64,
        n_selected: int = 16,
        conv_size: int = 4,
        n_experts: int | None = None,
        n_active: int = 2,
        d_expert: int | None = None,
        d_latent: int | None = None,
        expert_bank: ExpertBank | None = None,
    ):
        super().__init__()
        # `linear` picks the sequence mixer: a Gated DeltaNet-2 (linear attention,
        # recurrent, no RoPE) for the intra-block layers, or Native Sparse
        # Attention (sparse softmax, partial RoPE) for the block-tail global
        # layer. The two carry different cache formats at decode (a recurrent
        # state vs. a growing KV cache), so `linear` also routes decode().
        self.linear = linear
        self.mix_attn = DepthAttention(d_model)
        self.mix_ffn = DepthAttention(d_model)
        if linear:
            self.attn = GatedDeltaNet2(
                d_model, n_heads, n_kv_heads=n_kv_heads, conv_size=conv_size
            )
        else:
            # NSA keeps its own KV head count (nsa_kv_heads, default MQA):
            # selection is shared per GQA group and fla's kernels need the
            # group size (n_heads / nsa_kv_heads) to be a multiple of 16.
            self.attn = NativeSparseAttention(
                d_model,
                n_heads,
                n_kv_heads=nsa_kv_heads,
                block_size=sel_block,
                n_selected=n_selected,
                window=window_size,
                rope_factor=rope_factor,
                rope_base=rope_base,
                max_seq_len=max_seq_len,
            )
        if d_expert is None:
            d_expert = d_ffn
        self.ffn = SwiGLU(d_model, d_hidden=d_ffn)
        if n_experts is not None:
            self.moe = MixtureOfExperts(
                d_model,
                d_hidden=d_expert,
                n_experts=n_experts,
                n_active=n_active,
                d_latent=d_latent,
                bank=expert_bank,
            )

    def _mix(self, mixed: Tensor, doc_ids: Tensor | None, cu_seqlens: Tensor | None):
        # Dispatch the mixed input to the layer's sequence mixer: both reset
        # at cu_seqlens boundaries (GDN its recurrence, NSA its block/attention
        # structure); GDN additionally uses doc_ids for its short conv.
        if self.linear:
            return self.attn(mixed, cu_seqlens=cu_seqlens, doc_ids=doc_ids)
        return self.attn(mixed, cu_seqlens=cu_seqlens)

    def forward(
        self,
        blocks: Tensor,
        partial: Tensor | None,
        doc_ids: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor:
        # Block AttnRes protocol (see Transformer.forward): instead of adding
        # onto a single residual stream, each sublayer reads its input as depth
        # attention over the block representations plus the current block's
        # partial sum, and accumulates its output into that partial sum. The MoE
        # branch shares the FFN's mix: the two run in parallel from the same
        # input, forming one MLP sublayer.
        a = self._mix(self.mix_attn(blocks, partial), doc_ids, cu_seqlens)
        partial = a if partial is None else partial + a
        return self._mlp(blocks, partial, no_drop=False)

    def decode(
        self,
        blocks: Tensor,
        partial: Tensor | None,
        cache=None,
        pos: int = 0,
    ) -> tuple[Tensor, object]:
        mixed = self.mix_attn(blocks, partial)
        if self.linear:
            a, cache = self.attn.decode(mixed, cache)  # (rec_state, conv_state)
        else:
            a, cache = self.attn.decode(mixed, cache, pos)  # raw K/V dict
        partial = a if partial is None else partial + a
        # no_drop=True mirrors forward()'s MoE but never drops a token, so
        # generation runs the same network as training (see MoE.forward).
        return self._mlp(blocks, partial, no_drop=True), cache

    def _mlp(self, blocks: Tensor, partial: Tensor, no_drop: bool) -> Tensor:
        # The MLP sublayer, shared by forward()/decode(): the dense FFN and (if
        # present) the routed MoE run in parallel from the same depth-attention
        # mix and accumulate into the block's partial sum. MoE layers must apply
        # their experts at inference too, or generation would silently run a
        # different (FFN-only) network than training -- decode passes no_drop.
        h = self.mix_ffn(blocks, partial)
        if hasattr(self, "moe"):
            return partial + self.ffn(h) + self.moe(h, no_drop=no_drop)
        return partial + self.ffn(h)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_kv_heads: int | None = None,
        rope_base: float = 1_000_000.0,
        rope_factor: float = 0.25,
        d_ffn: int | None = None,
        max_seq_len: int = 4096,
        grad_checkpoint: bool = False,
        window_size: int = 512,
        layers_per_block: int = 4,
        nsa_kv_heads: int = 1,
        sel_block: int = 64,
        n_selected: int = 16,
        conv_size: int = 4,
        n_experts: int | None = None,
        d_expert: int | None = None,
        n_active: int = 2,
        d_latent: int | None = None,
        share_experts: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        # layers_per_block groups the layers into blocks that serve two roles at
        # once: the last layer of each block is a Native Sparse Attention
        # (global) layer while the rest are Gated DeltaNet-2 (linear) layers -- a
        # 3-GDN : 1-NSA hybrid at the default lpb=4 -- and each block is one unit
        # of the Block AttnRes residual (see DepthAttention / forward), committed
        # right after its NSA layer has integrated long-range context.
        # layers_per_block=1 makes every layer an NSA layer and its own block.
        self.layers_per_block = layers_per_block
        # Trade compute for memory during training: don't keep each layer's
        # activations for the backward pass, recompute them instead. Lets us fit
        # bigger models / longer sequences on a fixed GPU. No effect on decode().
        self.grad_checkpoint = grad_checkpoint

        self.layers = nn.ModuleList()
        # share_experts: every layer dispatches into one routed-expert pool
        # (MoEUT-style) while keeping its own router, load-balancing bias,
        # latent projections and dense FFN (the shared-expert analogue). The
        # first layer builds the bank through the normal default chain
        # (d_expert -> d_ffn -> 3*d_model); the rest receive that instance.
        expert_bank = None
        for i in range(n_layers):
            # block tail (every layers_per_block-th layer) -> NSA; rest -> GDN
            linear = (i + 1) % layers_per_block != 0
            layer = TransformerLayer(
                d_model,
                n_heads,
                linear,
                n_kv_heads=n_kv_heads,
                d_ffn=d_ffn,
                max_seq_len=max_seq_len,
                window_size=window_size,
                rope_base=rope_base,
                rope_factor=rope_factor,
                nsa_kv_heads=nsa_kv_heads,
                sel_block=sel_block,
                n_selected=n_selected,
                conv_size=conv_size,
                n_experts=n_experts,
                d_expert=d_expert,
                n_active=n_active,
                d_latent=d_latent,
                expert_bank=expert_bank,
            )
            if share_experts and n_experts is not None and expert_bank is None:
                expert_bank = layer.moe.bank
            self.layers.append(layer)
        # Final aggregation over all block representations before the head.
        self.mix_out = DepthAttention(d_model)

    def forward(
        self,
        x: Tensor,
        doc_ids: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor:
        # doc_ids (b, l): id of the packed document each token belongs to. When
        # given, the NSA layers mask attention within documents and the GDN
        # layers reset their recurrent state at document/row boundaries via
        # `cu_seqlens` -- derived here (eager convenience) when only doc_ids is
        # passed, or precomputed by the training modules outside the compiled
        # forward and passed in (both are traceable tensor inputs).
        if doc_ids is not None and cu_seqlens is None:
            cu_seqlens = doc_ids_to_cu_seqlens(doc_ids)
        # Block AttnRes state: `blocks` stacks the committed block
        # representations (starting with the token embedding as blocks[0]),
        # `partial` is the running sum of the current block's sublayer outputs.
        # Both thread through the layers as explicit args -- also across the
        # gradient-checkpoint boundary, where `blocks` grows by one tensor per
        # completed block but its rows are shared references, so checkpointing
        # keeps O(n_layers) distinct (b, t, d) activations alive as before.
        blocks, partial = x[None], None
        for i, layer in enumerate(self.layers):
            blocks, partial = self._open_block(blocks, partial, i)
            if self.grad_checkpoint and self.training:
                partial = torch.utils.checkpoint.checkpoint(
                    layer, blocks, partial, doc_ids, cu_seqlens, use_reentrant=False
                )
            else:
                partial = layer(blocks, partial, doc_ids, cu_seqlens)

        if self.training:
            # Apply each MoE's staged load-balancing bias update here -- once,
            # outside the checkpointed layer forwards above. Done inside
            # MoE.forward it would run twice under gradient checkpointing (see
            # MixtureOfExperts.forward).
            for layer in self.layers:
                if hasattr(layer, "moe"):
                    layer.moe.apply_bias_update()
        return self._finalize(blocks, partial)

    def _open_block(
        self, blocks: Tensor, partial: Tensor | None, i: int
    ) -> tuple[Tensor, Tensor | None]:
        # At a block boundary, commit the finished block's summed sublayer
        # outputs onto `blocks` and open a fresh (empty) partial; the next layer
        # (like every block's first sublayer) then attends over completed blocks
        # only. A no-op mid-block. Shared by forward()/decode().
        if i > 0 and i % self.layers_per_block == 0:
            return torch.cat([blocks, partial[None]]), None
        return blocks, partial

    def _finalize(self, blocks: Tensor, partial: Tensor | None) -> Tensor:
        # The head reads a final depth-attention aggregate of every block (the
        # last one possibly still partial) rather than the last partial sum.
        return rms_norm(self.mix_out(blocks, partial))

    def decode(
        self,
        x: Tensor,
        cache: list | None = None,
        pos: int = 0,
    ) -> tuple[Tensor, list, int]:
        # Owns the absolute-position bookkeeping in one place: every layer sees
        # the same `pos` (they all process the same chunk at the same time), and
        # only this method computes/advances it. Neither the cache nor the
        # position is kept as model state -- both flow through args/returns only.
        # The AttnRes blocks/partial state is per-token and lives only within
        # this call (rebuilt for each chunk). `cache[i]` is whatever the layer's
        # mixer returns: a Gated DeltaNet-2 (recurrent_state, conv_state) tuple, or
        # a Native Sparse Attention raw K/V dict.
        if cache is None:
            cache = [None] * self.n_layers
        q_len = x.shape[-2]
        blocks, partial = x[None], None
        for i, layer in enumerate(self.layers):
            blocks, partial = self._open_block(blocks, partial, i)
            partial, cache[i] = layer.decode(blocks, partial, cache[i], pos)
        return self._finalize(blocks, partial), cache, pos + q_len  # type: ignore


class MTPHead(nn.Module):
    """A parameter-light multi-token-prediction head (Medusa-style): a residual
    transform in d_model space whose output is decoded by the *shared* lm head,
    so the expensive vocab projection is not duplicated. Each head costs only
    d_model*d_model (or 2*d_model*rank when `rank` is set) instead of a full
    vocab*d_model output layer -- e.g. ~d_model/vocab of it.

    The transform is zero-initialized (see TransformerLM._init_weights), so at
    init `forward` returns the hidden state unchanged and the head predicts
    exactly like the primary lm head; training then specializes it to its offset.
    """

    def __init__(self, d_model: int, rank: int | None = None):
        super().__init__()
        # rank=None: one square transform. rank=r: a low-rank d_model->r->d_model
        # bottleneck (cheaper still when r < d_model/2).
        self.proj_in = None if rank is None else nn.Linear(d_model, rank, bias=False)
        self.out = nn.Linear(d_model if rank is None else rank, d_model, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        z = hidden if self.proj_in is None else self.proj_in(hidden)
        return hidden + F.gelu(self.out(z))  # residual; identity at init


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_kv_heads: int | None = None,
        rope_base: float = 1_000_000.0,
        rope_factor: float = 0.25,
        d_ffn: int | None = None,
        max_seq_len: int = 4096,
        init_std: float = 0.02,
        grad_checkpoint: bool = True,
        window_size: int = 512,
        layers_per_block: int = 4,
        nsa_kv_heads: int = 1,
        sel_block: int = 64,
        n_selected: int = 16,
        conv_size: int = 4,
        n_experts: int | None = None,
        n_active: int = 2,
        d_expert: int | None = None,
        d_latent: int | None = None,
        share_experts: bool = False,
        n_mtp: int = 0,
        mtp_rank: int | None = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.init_std = init_std

        self.embed = nn.Embedding(vocab_size, d_model)
        self.lmhead = nn.Linear(d_model, vocab_size, bias=False)
        # Multi-token prediction (Gloeckle et al., 2024), simplest "several output
        # heads" form: from the final hidden state h_t the primary lmhead predicts
        # token t+1, and mtp_heads[j] predicts token t+2+j (offsets +2..+(1+n_mtp)).
        # Each head is a light residual transform (MTPHead) decoded by the SHARED
        # lmhead, so it costs ~d_model^2 rather than a full vocab*d_model layer
        # (mtp_rank shrinks it further). Trained as an auxiliary loss (see
        # picochat.training) and used to draft tokens for self-speculative decoding
        # (picochat.inference.engine.generate_speculative). Plain autoregressive
        # decode ignores them (only the primary head runs). n_mtp=0 -> standard
        # model.
        self.n_mtp = n_mtp
        self.mtp_heads = nn.ModuleList(
            MTPHead(d_model, rank=mtp_rank) for _ in range(n_mtp)
        )
        self.transformer = Transformer(
            d_model,
            n_heads,
            n_layers,
            n_kv_heads=n_kv_heads,
            rope_base=rope_base,
            rope_factor=rope_factor,
            d_ffn=d_ffn,
            max_seq_len=max_seq_len,
            grad_checkpoint=grad_checkpoint,
            window_size=window_size,
            layers_per_block=layers_per_block,
            nsa_kv_heads=nsa_kv_heads,
            sel_block=sel_block,
            n_selected=n_selected,
            conv_size=conv_size,
            n_experts=n_experts,
            n_active=n_active,
            d_expert=d_expert,
            d_latent=d_latent,
            share_experts=share_experts,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # GPT-2 style: every weight ~ normal(0, init_std) with zero biases, then
        # the projections that write into the residual stream (attention/FFN/
        # expert outputs) are re-initialized with std scaled down by
        # 1/sqrt(2*n_layers) so the residual variance stays roughly constant with
        # depth.
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
        scaled_std = self.init_std / math.sqrt(2 * self.n_layers)
        for m in self.modules():
            if isinstance(m, (GatedDeltaNet2, NativeSparseAttention)):
                # both mixers write into the residual stream through proj_o
                nn.init.normal_(m.proj_o.weight, mean=0.0, std=scaled_std)
                if isinstance(m, GatedDeltaNet2):
                    # restore the gate parameters' non-normal init (the generic
                    # loop above only touched Linear/Embedding weights)
                    m.reset_parameters()
            elif isinstance(m, SwiGLU):
                nn.init.normal_(m.proj_down.weight, mean=0.0, std=scaled_std)
            elif isinstance(m, MixtureOfExperts):
                # With d_latent set, the shared expansion (not the experts'
                # down projection) is what writes into the residual stream.
                # (With share_experts the bank's weight_down is just re-drawn
                # once per owning layer -- same distribution, harmless.)
                if m.d_latent is not None:
                    nn.init.normal_(m.weight_expand, mean=0.0, std=scaled_std)
                else:
                    nn.init.normal_(m.bank.weight_down, mean=0.0, std=scaled_std)
        # MTP heads start as the identity (their residual transform outputs zero),
        # so at init each predicts exactly like the shared primary head; training
        # then specializes it to its offset. gelu'(0)=0.5, so gradients still flow.
        for head in self.mtp_heads:
            nn.init.zeros_(head.out.weight)

    def encode(
        self,
        x: Tensor | None = None,
        doc_ids: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        # Shared trunk (embed + transformer) up to the final hidden state, before
        # the lm head. doc_ids/cu_seqlens: see Transformer.forward (sequence
        # packing). `inputs_embeds` (B, L, d_model) bypasses the token embedding
        # so a caller can splice in non-text embeddings -- e.g. audio soft tokens
        # scattered over placeholder positions (see
        # picochat.model.audio.scatter_audio_embeds); when given, `x` is ignored.
        embeds = inputs_embeds if inputs_embeds is not None else self.embed(x)
        return self.transformer(embeds, doc_ids, cu_seqlens)

    def forward(
        self,
        x: Tensor | None = None,
        doc_ids: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        return self.lmhead(self.encode(x, doc_ids, cu_seqlens, inputs_embeds))

    def _decode_trunk(
        self,
        x: Tensor | None,
        cache: list | None,
        pos: int,
        inputs_embeds: Tensor | None,
    ) -> tuple[Tensor, list, int]:
        # Shared decode trunk (embed + cached transformer) up to the final hidden
        # states. `inputs_embeds` prefills the cache from spliced embeddings
        # (e.g. an audio-conditioned prompt) instead of token ids; later steps
        # pass token ids as usual.
        embeds = inputs_embeds if inputs_embeds is not None else self.embed(x)
        return self.transformer.decode(embeds, cache, pos)

    def decode(
        self,
        x: Tensor | None = None,
        cache: list | None = None,
        pos: int = 0,
        inputs_embeds: Tensor | None = None,
    ) -> tuple[Tensor, list, int]:
        embeds, cache, pos = self._decode_trunk(x, cache, pos, inputs_embeds)
        return self.lmhead(embeds), cache, pos

    def decode_heads(
        self,
        x: Tensor | None = None,
        cache: list | None = None,
        pos: int = 0,
        inputs_embeds: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list, int]:
        """Like decode() but also returns every MTP head's logits, for
        speculative drafting. Returns (logits, mtp_logits, cache, pos), where
        logits is (B, q_len, vocab) from the primary head and mtp_logits is a
        length-n_mtp list of (B, q_len, vocab) tensors (head j -> offset 2+j).
        Chunked decode is per-position equivalent to sequential decode (attention
        is causal, MoE runs no_drop), so a whole candidate chunk can be verified
        in one call. The caller snapshots the cache before drafting so rejected
        drafts can be rolled back (see picochat.inference.engine
        .generate_speculative)."""
        embeds, cache, pos = self._decode_trunk(x, cache, pos, inputs_embeds)
        # each MTP head transforms the hidden state, then the SHARED lm head
        # decodes it to vocab logits.
        logits = self.lmhead(embeds)
        mtp = [self.lmhead(head(embeds)) for head in self.mtp_heads]
        return logits, mtp, cache, pos
