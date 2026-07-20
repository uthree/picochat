"""The picochat model (nanochat-style): the building blocks (norm, SwiGLU, MoE,
depth-attention residuals) up through TransformerLM. The two sequence mixers
live in their own modules -- Gated DeltaNet (linear attention) in linear_attn.py
and Native Sparse Attention in sparse_attn.py -- and are interleaved here (3 GDN
: 1 NSA per block). The scale-ladder presets and the build_lm factory live in
presets.py, the parameter estimator in param_estimate.py; the LightningModules
that train it live in trainer.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from picochat.linear_attn import GatedDeltaNet
from picochat.sparse_attn import NativeSparseAttention


def rms_norm(x: Tensor, eps: float = 1e-8) -> Tensor:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        return x / (x.square().mean(-1, keepdim=True).sqrt() + eps)


def doc_ids_to_cu_seqlens(doc_ids: Tensor) -> Tensor:
    """Segment boundaries (into the flattened batch*seq layout) at which a Gated
    DeltaNet layer must reset its recurrent state: every document boundary within
    a packed row, plus every row boundary (so flattening rows never lets the
    recurrence leak across them). Returns a 1D LongTensor starting at 0 and
    ending at batch*seq. Data-dependent, so built outside the compiled forward
    (like the old packed_masks)."""
    b, seq = doc_ids.shape
    flat = doc_ids.clone()
    # make doc ids unique per row so a row boundary always reads as a change
    flat = flat + (torch.arange(b, device=doc_ids.device)[:, None] * (doc_ids.max() + 1))
    flat = flat.reshape(-1)
    change = torch.ones(b * seq, dtype=torch.bool, device=doc_ids.device)
    change[1:] = flat[1:] != flat[:-1]
    bounds = torch.nonzero(change, as_tuple=False).squeeze(-1)
    return torch.cat([bounds, bounds.new_tensor([b * seq])])


class SwiGLU(nn.Module):
    def __init__(
        self, d_model: int, d_hidden: int | None = None, p_dropout: float = 0.1
    ):
        super().__init__()
        self.p_dropout = p_dropout
        if d_hidden is None:
            d_hidden = d_model * 3
        self.proj_up = nn.Linear(d_model, d_hidden, bias=False)
        self.proj_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.proj_down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = rms_norm(x)
        x = self.proj_up(x) * F.silu(self.proj_gate(x))
        x = F.dropout(x, self.p_dropout, training=self.training)
        x = self.proj_down(x)
        x = F.dropout(x, self.p_dropout, training=self.training)
        return x


class ExpertBank(nn.Module):
    """The fused routed-expert weights of a MixtureOfExperts, factored out so
    they can be owned per layer (the default) or built once and shared by
    every layer's MoE (Transformer's share_experts, MoEUT-style): each layer
    then keeps its own router, load-balancing bias and latent projections
    while dispatching into the same expert pool.

    The weights are stored 2D, experts stacked along the rows --
    (n_experts * out_features, in_features) -- and viewed back to
    (n_experts, out, in) for the bmm in forward. torch.optim.Muon accepts
    only 2D parameters, and this flattened matrix is exactly what Muon
    orthogonalizes for a stacked expert weight anyway.

    A shared bank is registered as a submodule of every owning MoE:
    parameters()/modules() deduplicate it, torch.save stores the shared
    storage once, and load_state_dict writes the same values through each
    duplicate key -- harmless in both directions.
    """

    def __init__(self, n_experts: int, d_io: int, d_hidden: int):
        super().__init__()
        self.n_experts = n_experts
        self.d_io = d_io  # d_model, or d_latent for a LatentMoE
        self.d_hidden = d_hidden
        self.weight_up = nn.Parameter(torch.empty(n_experts * d_hidden, d_io))
        self.weight_gate = nn.Parameter(torch.empty(n_experts * d_hidden, d_io))
        self.weight_down = nn.Parameter(torch.empty(n_experts * d_io, d_hidden))
        for w in (self.weight_up, self.weight_gate, self.weight_down):
            nn.init.normal_(w, mean=0.0, std=0.02)

    def forward(self, expert_in: Tensor, p_dropout: float = 0.0) -> Tensor:
        # expert_in (n_experts, capacity, d_io): each expert's dispatched
        # tokens; runs every expert's SwiGLU as one batched bmm.
        w_up = self.weight_up.view(self.n_experts, self.d_hidden, self.d_io)
        w_gate = self.weight_gate.view(self.n_experts, self.d_hidden, self.d_io)
        w_down = self.weight_down.view(self.n_experts, self.d_io, self.d_hidden)
        up = torch.bmm(expert_in, w_up.transpose(1, 2))
        gate = torch.bmm(expert_in, w_gate.transpose(1, 2))
        h = F.dropout(up * F.silu(gate), p_dropout, training=self.training)
        out = torch.bmm(h, w_down.transpose(1, 2))
        return F.dropout(out, p_dropout, training=self.training)


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        p_dropout: float = 0.1,
        n_experts: int = 8,
        n_active: int = 2,
        capacity_factor: float = 1.25,
        bias_update_rate: float = 1e-3,
        d_latent: int | None = None,
        bank: ExpertBank | None = None,
    ):
        super().__init__()
        assert n_active <= n_experts
        self.p_dropout = p_dropout
        self.n_experts = n_experts
        self.n_active = n_active
        # capacity_factor: slack over the perfectly-even per-expert token count
        # (see forward). bias_update_rate: how fast expert_bias below chases
        # load balance.
        self.capacity_factor = capacity_factor
        self.bias_update_rate = bias_update_rate
        if d_hidden is None:
            d_hidden = d_model * 3
        self.d_hidden = d_hidden
        # LatentMoE (arXiv:2601.18089): when d_latent is set, tokens are
        # compressed to that dimension by a shared down-projection before
        # dispatch, the routed experts run their SwiGLU entirely in the latent
        # space (the intermediate d_hidden stays as configured, preserving the
        # nonlinear budget), and the combined output is expanded back to
        # d_model by a shared up-projection. Expert weight loading and token
        # dispatch then cost d_latent instead of d_model each, the saving the
        # paper reinvests in more experts / higher top-k (a preset choice, via
        # n_experts / n_active). The router (below) and the dense FFN that
        # TransformerLayer runs in parallel (the paper's shared expert) stay
        # in d_model. d_latent=None keeps the standard MoE unchanged.
        self.d_latent = d_latent
        d_expert_io = d_model if d_latent is None else d_latent
        self.d_expert_io = d_expert_io
        self.weight_router = nn.Parameter(torch.empty(n_experts, d_model))
        # The routed-expert weights live in an ExpertBank (see there) so they
        # can be shared across layers: with a shared bank passed in, this MoE
        # owns only its router, load-balancing bias and latent projections.
        if bank is None:
            bank = ExpertBank(n_experts, d_expert_io, d_hidden)
        assert bank.n_experts == n_experts
        assert bank.d_io == d_expert_io and bank.d_hidden == d_hidden
        self.bank = bank
        weights = [self.weight_router]
        if d_latent is not None:
            self.weight_compress = nn.Parameter(torch.empty(d_latent, d_model))
            self.weight_expand = nn.Parameter(torch.empty(d_model, d_latent))
            weights += [self.weight_compress, self.weight_expand]
        for w in weights:
            nn.init.normal_(w, mean=0.0, std=0.02)
        # Normalize the aggregated (routed) expert output before it re-enters the
        # residual stream. The summed contribution of the selected experts has a
        # scale that drifts with routing, while fine-grained MoE outputs come out
        # systematically scaled down versus a dense FFN. A learnable RMSNorm here
        # rescales it and stabilizes training; it gives a small gain for standard
        # MoE but is crucial for fine-grained experts (Scaling Laws for
        # Fine-Grained MoE, arXiv:2402.07871), the same output normalization
        # DeepSeek-V3 / Kimi-style stacks apply. Runs on the d_model output (after
        # the latent expansion when d_latent is set).
        self.out_gain = nn.Parameter(torch.ones(d_model))
        # DeepSeek-V3 style aux-loss-free load balancing: a per-expert bias
        # that only steers *which* experts get picked (added before top-k,
        # dropped again before computing combine weights below), nudged every
        # training step toward under-loaded experts. Not a Parameter -- no
        # gradient, no loss term, just a running buffer.
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        # The per-expert load counts are *staged* here in forward and turned
        # into a bias update once by Transformer.forward, outside any
        # gradient-checkpoint boundary -- see forward() / apply_bias_update().
        # Non-persistent: derived per-step, not part of the checkpointed state.
        self.register_buffer(
            "_pending_load", torch.zeros(n_experts), persistent=False
        )

    def forward(self, x: Tensor, no_drop: bool = False) -> Tensor:
        b, t, d = x.shape
        n_tokens = b * t
        tokens = rms_norm(x).reshape(n_tokens, d)

        # Route every token to its top-n_active experts (DeepSeek-V3 style). The
        # router affinity is a per-expert sigmoid; expert_bias only steers *which*
        # experts are picked (aux-loss-free load balancing), while the combine
        # weights are the selected affinities renormalized to sum to 1, so they
        # stay a differentiable function of weight_router alone.
        scores = torch.sigmoid(tokens @ self.weight_router.T)  # (n_tokens, n_experts)
        top_idx = (scores + self.expert_bias).topk(self.n_active, dim=-1).indices
        top_scores = scores.gather(-1, top_idx)  # (T, n_active)
        top_weight = top_scores / top_scores.sum(-1, keepdim=True).clamp_min(1e-9)

        # Flatten the (token, expert) assignments to one row per routed pair, in
        # slot-major order so a token's 1st-choice expert claims capacity before
        # its 2nd choice.
        pair_expert = top_idx.T.reshape(-1)  # (n_active * T,)
        pair_token = torch.arange(n_tokens, device=x.device).repeat(self.n_active)
        pair_weight = top_weight.T.reshape(-1)

        # Give each pair a slot within its expert (its rank among pairs routed
        # there). Fixed per-expert capacity keeps every tensor statically shaped
        # -- traceable under torch.compile, unlike a data-dependent gather -- and
        # pairs past capacity are dropped (Switch Transformer / GShard style).
        counts = F.one_hot(pair_expert, self.n_experts)  # (pairs, n_experts)
        slot = (counts.cumsum(0) - 1).gather(-1, pair_expert[:, None]).squeeze(-1)
        if no_drop:
            # Decode: size capacity so nothing overflows (a token routes to each
            # expert at most once, so no expert gets more than n_tokens pairs).
            # Makes the layer purely per-token, so chunked decode matches a full
            # forward exactly. Only reached from decode(), where n_tokens is small.
            capacity = n_tokens
        else:
            capacity = max(
                self.n_active,
                math.ceil(
                    n_tokens * self.n_active * self.capacity_factor / self.n_experts
                ),
            )
        keep = slot < capacity
        n_slots = self.n_experts * capacity
        # Dropped pairs are redirected to a trash-bin row (index n_slots) so the
        # scatter/gather below use a fixed-size index instead of a dynamic one.
        dest = torch.where(keep, pair_expert * capacity + slot, n_slots)

        if self.training:
            # Stage the raw load counts rather than updating the bias here.
            # Under gradient checkpointing this forward runs twice (once for
            # real, once recomputed in backward); staging into a buffer is
            # idempotent, and Transformer.forward turns it into exactly one
            # bias update outside the checkpoint (where the counts are also
            # summed across DDP ranks -- see apply_bias_update).
            with torch.no_grad():
                # tokens routed to each expert
                self._pending_load.copy_(counts.sum(0).float())

        # Dispatch pairs into per-expert buffers, run the expert SwiGLU, combine.
        # With d_latent set (LatentMoE), dispatch, expert computation and the
        # combine all happen in the latent dimension d_io; only the final
        # expansion returns to d_model. Weighting the contributions before the
        # (linear) expansion matches the paper's W_up(sum p_i * E_i) exactly.
        d_io = self.d_expert_io
        inputs = tokens if self.d_latent is None else tokens @ self.weight_compress.T
        keep_f = keep.unsqueeze(-1).to(tokens.dtype)
        buffer = inputs.new_zeros(n_slots + 1, d_io).index_add(
            0, dest, inputs[pair_token] * keep_f
        )
        expert_in = buffer[:n_slots].reshape(self.n_experts, capacity, d_io)
        expert_out = self.bank(expert_in, self.p_dropout)
        expert_out = torch.cat(
            [expert_out.reshape(n_slots, d_io), expert_out.new_zeros(1, d_io)], dim=0
        )

        contrib = expert_out[dest] * (pair_weight.unsqueeze(-1) * keep_f)
        out = inputs.new_zeros(n_tokens, d_io).index_add(0, pair_token, contrib)
        if self.d_latent is not None:
            out = out @ self.weight_expand.T
        out = rms_norm(out) * self.out_gain  # normalize the aggregated output
        return out.reshape(b, t, d)

    @torch.no_grad()
    def apply_bias_update(self) -> None:
        # Turn the load counts staged by the most recent forward into one bias
        # step toward under-loaded experts. Called by Transformer.forward once
        # per step, outside the gradient-checkpoint boundary, so the update
        # lands exactly once whether or not the layer was recomputed in
        # backward. Under DDP the counts are summed across ranks first: the
        # bias then chases the *global* batch's load (as DeepSeek-V3 does) and
        # stays identical on every rank instead of following rank 0's local
        # batch only (DDP broadcasts buffers from rank 0 each forward, which
        # would silently discard the other ranks' updates otherwise).
        load = self._pending_load
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            load = load.clone()
            torch.distributed.all_reduce(load)
        self.expert_bias += self.bias_update_rate * torch.sign(load.mean() - load)


class DepthAttention(nn.Module):
    """Depth-wise softmax attention over block representations -- the residual
    mixing of Block Attention Residuals (arXiv:2603.15031). Replaces the
    fixed-unit-weight residual stream: a sublayer's input is a softmax-weighted
    mix over the token embedding, every completed block representation and the
    current block's partial sum, weighted by a learned per-sublayer query
    against RMSNorm'd keys. The norm keeps blocks with naturally larger
    magnitudes (sums over more sublayers) from dominating the weights; the
    values themselves stay unnormalized. The zero-init query starts every
    sublayer at a uniform mix, and softmax (not sigmoid) provides the
    competitive selection the paper found essential. Per-token and
    mask-independent, so sequence packing, sliding windows and the KV cache
    are unaffected.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 1-dim on purpose: lands in the AdamW no-decay group, not Muon (see
        # trainer._muon_param_split).
        self.query = nn.Parameter(torch.zeros(d_model))

    def forward(self, blocks: Tensor, partial: Tensor | None) -> Tensor:
        # blocks (n, b, t, d): committed block representations, blocks[0] being
        # the token embedding. partial (b, t, d): the current block's running
        # sum of sublayer outputs; None at a block's first sublayer, where only
        # completed blocks are visible.
        values = blocks if partial is None else torch.cat([blocks, partial[None]])
        weight = (rms_norm(values) * self.query).sum(-1).softmax(0)
        return (values * weight[..., None]).sum(0)


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
        # `linear` picks the sequence mixer: a Gated DeltaNet (linear attention,
        # recurrent, no RoPE) for the intra-block layers, or Native Sparse
        # Attention (sparse softmax, partial RoPE) for the block-tail global
        # layer. The two carry different cache formats at decode (a recurrent
        # state vs. a growing KV cache), so `linear` also routes decode().
        self.linear = linear
        self.mix_attn = DepthAttention(d_model)
        self.mix_ffn = DepthAttention(d_model)
        if linear:
            self.attn = GatedDeltaNet(
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
        # (global) layer while the rest are Gated DeltaNet (linear) layers -- a
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
        # mixer returns: a Gated DeltaNet (recurrent_state, conv_state) tuple, or
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
        # LMTrainerMixin) and used to draft tokens for self-speculative decoding
        # (picochat.engine.generate_speculative). Plain autoregressive decode
        # ignores them (only the primary head runs). n_mtp=0 -> standard model.
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
            if isinstance(m, (GatedDeltaNet, NativeSparseAttention)):
                # both mixers write into the residual stream through proj_o
                nn.init.normal_(m.proj_o.weight, mean=0.0, std=scaled_std)
                if isinstance(m, GatedDeltaNet):
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
        # picochat.audio.scatter_audio_embeds); when given, `x` is ignored.
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
        drafts can be rolled back (see picochat.engine.generate_speculative)."""
        embeds, cache, pos = self._decode_trunk(x, cache, pos, inputs_embeds)
        # each MTP head transforms the hidden state, then the SHARED lm head
        # decodes it to vocab logits.
        logits = self.lmhead(embeds)
        mtp = [self.lmhead(head(embeds)) for head in self.mtp_heads]
        return logits, mtp, cache, pos


def moe_modules(model: nn.Module) -> list[MixtureOfExperts]:
    """Every routed-expert layer in the model (empty list for a dense model)."""
    return [m for m in model.modules() if isinstance(m, MixtureOfExperts)]


