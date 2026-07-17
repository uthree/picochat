"""The picochat model, in one file (nanochat-style): the building blocks
(norm, RoPE, SwiGLU, MoE, attention) up through TransformerLM, plus the
scale-ladder presets and the build_lm/estimate_num_params helpers that size
it. The LightningModules that train it live in trainer.py.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


def rms_norm(x: Tensor, eps: float = 1e-8) -> Tensor:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        return x / (x.square().mean(-1, keepdim=True).sqrt() + eps)


_block_mask_cache: dict[tuple, BlockMask] = {}


@torch._dynamo.disable()
def _sliding_window_block_mask(
    window_size: int, q_len: int, k_len: int, device: torch.device
) -> BlockMask:
    # Building a BlockMask involves plain-Python/vmap machinery that dynamo
    # can't trace through, and doing so from inside a checkpointed layer (a
    # graph break inside a for-loop) makes torch.compile give up on the whole
    # forward and fall back to eager. `torch._dynamo.disable` keeps this call
    # opaque to the tracer so the surrounding flex_attention call still gets
    # compiled into a fused kernel; caching means it only actually runs once
    # per (window_size, q_len, k_len, device) instead of every forward call.
    key = (window_size, q_len, k_len, str(device))
    if key not in _block_mask_cache:

        def mask_mod(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & (kv_idx > q_idx - window_size)

        _block_mask_cache[key] = create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=k_len, device=device
        )
    return _block_mask_cache[key]


@torch._dynamo.disable()
def _packed_attention_mask(
    doc_ids: Tensor, window_size: int | None
) -> BlockMask | Tensor:
    """Causal mask that keeps attention within one packed document
    (MosaicBERT-style sequence packing): several documents share one training
    sequence, and a token may only attend to earlier tokens carrying the same
    doc id (further limited to the sliding window when window_size is set).

    Unlike _sliding_window_block_mask this is data-dependent (doc_ids changes
    every batch), so it can't be cached; Transformer.packed_masks builds it
    once per step per distinct window size. CUDA gets a flex_attention
    BlockMask (block-granular, cheap relative to the attention itself);
    elsewhere a dense (b, 1, l, l) bool mask for SDPA, since flex_attention
    has no CPU backward. RoPE is relative, so masking is all packing needs --
    no per-document position reset.

    dynamo-disabled because create_block_mask isn't traceable; but note that
    calling this *inside* a compiled forward silently drops the surrounding
    layers out of torch.compile once the returned BlockMask sits in a
    container across the graph break, which runs flex_attention unfused (it
    materializes the full scores matrix). The training modules therefore call
    packed_masks() outside the compiled callable and pass the masks in as
    inputs (BlockMask is a pytree, so it traces cleanly as a graph input);
    the in-forward fallback below is for eager/CPU use only.
    """
    if doc_ids.is_cuda:

        def mask_mod(b, h, q_idx, kv_idx):
            ok = (kv_idx <= q_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])
            if window_size is not None:
                ok = ok & (kv_idx > q_idx - window_size)
            return ok

        batch, seq_len = doc_ids.shape
        return create_block_mask(
            mask_mod,
            B=batch,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=doc_ids.device,
        )
    idx = torch.arange(doc_ids.shape[-1], device=doc_ids.device)
    mask = (idx[:, None] >= idx[None, :]) & (doc_ids[:, :, None] == doc_ids[:, None, :])
    if window_size is not None:
        mask &= idx[None, :] > idx[:, None] - window_size
    return mask[:, None]  # (b, 1, l, l): broadcasts over heads


def rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = torch.stack([-x2, x1], dim=-1)
    x = rearrange(x, "... d r -> ... (d r)")
    return x


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
        self.weight_router = nn.Parameter(torch.empty(n_experts, d_model))
        # The fused expert weights are stored 2D, experts stacked along the
        # rows -- (n_experts * out_features, in_features) -- and viewed back
        # to (n_experts, out, in) for the bmm in forward. torch.optim.Muon
        # accepts only 2D parameters, and this flattened matrix is exactly
        # what Muon orthogonalizes for a stacked expert weight anyway.
        self.weight_up = nn.Parameter(torch.empty(n_experts * d_hidden, d_model))
        self.weight_gate = nn.Parameter(torch.empty(n_experts * d_hidden, d_model))
        self.weight_down = nn.Parameter(torch.empty(n_experts * d_model, d_hidden))
        for w in (
            self.weight_router,
            self.weight_up,
            self.weight_gate,
            self.weight_down,
        ):
            nn.init.normal_(w, mean=0.0, std=0.02)
        # DeepSeek-V3 style aux-loss-free load balancing: a per-expert bias
        # that only steers *which* experts get picked (added before top-k,
        # dropped again before computing combine weights below), nudged every
        # training step toward under-loaded experts. Not a Parameter -- no
        # gradient, no loss term, just a running buffer.
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        # The load-balancing update is *staged* here in forward and applied once
        # by Transformer.forward, outside any gradient-checkpoint boundary --
        # see forward() / apply_bias_update(). Non-persistent: derived per-step,
        # not part of the checkpointed state.
        self.register_buffer(
            "_pending_bias_delta", torch.zeros(n_experts), persistent=False
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
            # Stage the load-balancing delta rather than applying it here. Under
            # gradient checkpointing this forward runs twice (once for real, once
            # recomputed in backward); staging into a buffer is idempotent, and
            # Transformer.forward applies it exactly once outside the checkpoint.
            with torch.no_grad():
                load = counts.sum(0).float()  # tokens routed to each expert
                self._pending_bias_delta.copy_(
                    self.bias_update_rate * torch.sign(load.mean() - load)
                )

        # Dispatch pairs into per-expert buffers, run the expert SwiGLU, combine.
        keep_f = keep.unsqueeze(-1).to(tokens.dtype)
        buffer = tokens.new_zeros(n_slots + 1, d).index_add(
            0, dest, tokens[pair_token] * keep_f
        )
        expert_in = buffer[:n_slots].reshape(self.n_experts, capacity, d)
        w_up = self.weight_up.view(self.n_experts, self.d_hidden, d)
        w_gate = self.weight_gate.view(self.n_experts, self.d_hidden, d)
        w_down = self.weight_down.view(self.n_experts, d, self.d_hidden)
        up = torch.bmm(expert_in, w_up.transpose(1, 2))
        gate = torch.bmm(expert_in, w_gate.transpose(1, 2))
        h = F.dropout(up * F.silu(gate), self.p_dropout, training=self.training)
        expert_out = torch.bmm(h, w_down.transpose(1, 2))
        expert_out = F.dropout(expert_out, self.p_dropout, training=self.training)
        expert_out = torch.cat(
            [expert_out.reshape(n_slots, d), expert_out.new_zeros(1, d)], dim=0
        )

        contrib = expert_out[dest] * (pair_weight.unsqueeze(-1) * keep_f)
        out = tokens.new_zeros(n_tokens, d).index_add(0, pair_token, contrib)
        return out.reshape(b, t, d)

    @torch.no_grad()
    def apply_bias_update(self) -> None:
        # Apply the load-balancing delta staged by the most recent forward (see
        # forward). Called by Transformer.forward once per step, outside the
        # gradient-checkpoint boundary, so the update lands exactly once whether
        # or not the layer was recomputed in backward.
        self.expert_bias += self._pending_bias_delta


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        rope_base: int = 10000,
        max_seq_len: int = 4096,
        window_size: int | None = None,  # If None is given, full attention
    ):
        super().__init__()
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        # Specify the number of query heads; the per-head dim is derived so proj_q
        # stays square (d_head * n_heads == d_model). GQA is set by n_kv_heads.
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        assert d_model % n_heads == 0  # heads tile d_model
        assert n_heads % self.n_kv_heads == 0  # GQA grouping
        assert self.d_head % 2 == 0  # RoPE rotates dimension pairs

        self.proj_q = nn.Linear(d_model, self.d_head * n_heads, bias=False)
        self.proj_k = nn.Linear(d_model, self.d_head * self.n_kv_heads, bias=False)
        self.proj_v = nn.Linear(d_model, self.d_head * self.n_kv_heads, bias=False)
        # Gated attention (arXiv:2505.06708, G1 variant): an input-dependent
        # elementwise sigmoid gate on the attention output, per query head,
        # applied before proj_o. Adds non-linearity between the value/output
        # projections and lets a head cleanly zero its contribution, which the
        # paper shows removes attention sinks and improves stability/quality.
        self.proj_g = nn.Linear(d_model, self.d_head * n_heads, bias=False)
        self.proj_o = nn.Linear(self.d_head * n_heads, d_model, bias=False)

        sin, cos = self._rope_tables(max_seq_len)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Shared q/k/v/gate projection (with QK-norm) for both forward and
        # decode. The gate is computed from the same normalized input as q/k/v
        # and multiplies the attention output elementwise in _output.
        x = rms_norm(x)
        query = rearrange(self.proj_q(x), "b l (h d) -> b h l d", d=self.d_head)
        key = rms_norm(rearrange(self.proj_k(x), "b l (g d) -> b g l d", d=self.d_head))
        value = rms_norm(
            rearrange(self.proj_v(x), "b l (g d) -> b g l d", d=self.d_head)
        )
        gate = torch.sigmoid(self.proj_g(x))  # (b, l, h*d)
        return query, key, value, gate

    def _output(self, attn: Tensor, gate: Tensor) -> Tensor:
        # Gate the attention output elementwise, then project back to d_model.
        return self.proj_o(rearrange(attn, "b h l d -> b l (h d)") * gate)

    def _window_mask(
        self,
        q_len: int,
        k_len: int,
        q_offset: int,
        k_offset: int,
        device: torch.device,
    ) -> Tensor:
        # Bottom-right aligned causal mask: query at absolute position i+q_offset
        # attends to keys at absolute position <= i+q_offset (reduces to plain
        # causal when both offsets are 0). Keys need their own offset separately
        # from queries because a truncated KV cache no longer starts at absolute
        # position 0 (see SelfAttention.decode). When window_size is set,
        # additionally drop keys older than window_size positions back, so each
        # query only sees a local trailing slice.
        q_idx = torch.arange(q_len, device=device).unsqueeze(1) + q_offset
        k_idx = torch.arange(k_len, device=device).unsqueeze(0) + k_offset
        mask = k_idx <= q_idx
        if self.window_size is not None:
            mask &= k_idx > q_idx - self.window_size
        return mask

    def forward(self, x: Tensor, attn_mask: BlockMask | Tensor | None = None) -> Tensor:
        # Training path: causal attention over the whole sequence, no cache.
        # `attn_mask` is the packed-document mask built by Transformer.forward
        # (a flex_attention BlockMask on CUDA, a dense bool mask elsewhere); it
        # already encodes causality, document boundaries and this layer's
        # window, so it replaces the plain causal/windowed paths below.
        query, key, value, gate = self._project(x)
        query, key = self._rope(query), self._rope(key)
        if isinstance(attn_mask, BlockMask):
            attn = flex_attention(
                query, key, value, block_mask=attn_mask, enable_gqa=True
            )
        elif attn_mask is not None:
            attn = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, enable_gqa=True
            )
        elif self.window_size is None:
            # Fast path: let SDPA use its native causal kernel instead of a
            # materialized mask.
            attn = F.scaled_dot_product_attention(
                query, key, value, is_causal=True, enable_gqa=True
            )
        elif query.is_cuda:
            # flex_attention lowers to a fused, block-sparse Triton kernel (the
            # same flash-attention family of algorithms as SDPA's fused
            # backends) when this forward runs under torch.compile, so windowed
            # layers skip whole blocks outside the window instead of
            # materializing an L x L mask like a naive implementation would.
            # (flex_attention has no CPU backward support, so this path is
            # CUDA-only; see the else branch below.)
            block_mask = _sliding_window_block_mask(
                self.window_size, query.shape[-2], key.shape[-2], query.device
            )
            attn = flex_attention(
                query, key, value, block_mask=block_mask, enable_gqa=True
            )
        else:
            mask = self._window_mask(
                query.shape[-2],
                key.shape[-2],
                q_offset=0,
                k_offset=0,
                device=query.device,
            )
            attn = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, enable_gqa=True
            )
        return self._output(attn, gate)

    def decode(
        self, x: Tensor, cache: Tensor | None = None, pos: int = 0
    ) -> tuple[Tensor, Tensor]:
        # Inference path: append the new keys/values to the cache and attend over
        # the full prefix (the window mask below limits which of those the query
        # actually attends to). `pos` is the absolute position of the first token
        # of `x`; the caller (Transformer.decode) owns this bookkeeping, since the
        # cache is truncated below and can no longer be used to infer it.
        # Always runs eager (see GPT.__init__), so flex_attention would gain
        # nothing here over a plain masked SDPA call; keep the simpler path.
        query, key, value, gate = self._project(x)
        old_len = 0 if cache is None else cache.shape[-2]
        if cache is not None:
            key = torch.cat([cache[0], key], dim=-2)
            value = torch.cat([cache[1], value], dim=-2)
        # Absolute position of the first (untruncated) key involved in this call.
        key_offset = pos - old_len
        q_len, k_len = query.shape[-2], key.shape[-2]
        query_r, key_r = (
            self._rope(query, offset=pos),
            self._rope(key, offset=key_offset),
        )
        mask = self._window_mask(
            q_len, k_len, q_offset=pos, k_offset=key_offset, device=query.device
        )
        attn = F.scaled_dot_product_attention(
            query_r, key_r, value, attn_mask=mask, enable_gqa=True
        )
        out = self._output(attn, gate)
        # Truncate only what's carried into the *next* call's cache; the attention
        # above already used the full, untruncated key/value, so a chunk longer
        # than window_size (e.g. a long prefill) is handled correctly for free.
        if self.window_size is not None:
            key = key[..., -self.window_size :, :]
            value = value[..., -self.window_size :, :]
        new_cache = torch.stack([key, value])
        return out, new_cache

    def _rope_tables(self, max_seq_len: int) -> tuple[Tensor, Tensor]:
        # Build sin/cos for absolute positions 0..max_seq_len-1 (offset is handled
        # later when slicing).
        t = torch.arange(max_seq_len)[:, None].float()
        f = (
            self.rope_base
            ** (torch.linspace(0.0, 1.0, self.d_head // 2).repeat_interleave(2))
        )[None, :]
        theta = t / f
        return torch.sin(theta), torch.cos(theta)

    def _rope(self, x, offset: int = 0) -> Tensor:
        seq_len = x.shape[-2]
        assert offset + seq_len <= self.max_seq_len, (
            f"position {offset + seq_len} exceeds max_seq_len={self.max_seq_len}"
        )
        # Apply RoPE in float32 to keep positional precision under bf16 autocast.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            sin = self.sin[offset : offset + seq_len, :]
            cos = self.cos[offset : offset + seq_len, :]
            x = x * cos + rotate_half(x) * sin
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        rope_base: int = 10000,
        d_ffn: int | None = None,
        max_seq_len: int = 4096,
        window_size: int | None = None,
        n_experts: int | None = None,
        n_active: int = 2,
        d_expert: int | None = None,
    ):
        super().__init__()
        self.attn = SelfAttention(
            d_model,
            n_heads,
            n_kv_heads=n_kv_heads,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
            window_size=window_size,
        )
        if d_expert is None:
            d_expert = d_ffn
        self.ffn = SwiGLU(d_model, d_hidden=d_ffn)
        if n_experts is not None:
            self.moe = MixtureOfExperts(
                d_model, d_hidden=d_expert, n_experts=n_experts, n_active=n_active
            )

    def forward(self, x: Tensor, attn_mask: BlockMask | Tensor | None = None) -> Tensor:
        # attn/ffn apply pre-norm (rms_norm) internally, so add the raw residual.
        x = self.attn(x, attn_mask) + x
        if hasattr(self, "moe"):
            x = self.ffn(x) + self.moe(x) + x
        else:
            x = self.ffn(x) + x
        return x

    def decode(
        self, x: Tensor, cache: Tensor | None = None, pos: int = 0
    ) -> tuple[Tensor, Tensor]:
        a, cache = self.attn.decode(x, cache, pos)
        x = a + x
        # Mirror forward(): MoE layers must apply their experts at inference too,
        # otherwise generation silently runs a different (FFN-only) network than
        # training. no_drop=True: never drop a token while generating.
        if hasattr(self, "moe"):
            x = self.ffn(x) + self.moe(x, no_drop=True) + x
        else:
            x = self.ffn(x) + x
        return x, cache


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_kv_heads: int | None = None,
        rope_base: int = 10000,
        d_ffn: int | None = None,
        max_seq_len: int = 4096,
        grad_checkpoint: bool = False,
        window_size: int = 64,
        global_attn_ratio: int = 4,
        n_experts: int | None = None,
        d_expert: int | None = None,
        n_active: int = 2,
    ):
        super().__init__()
        self.n_layers = n_layers
        # Trade compute for memory during training: don't keep each layer's
        # activations for the backward pass, recompute them instead. Lets us fit
        # bigger models / longer sequences on a fixed GPU. No effect on decode().
        self.grad_checkpoint = grad_checkpoint

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = TransformerLayer(
                d_model,
                n_heads,
                n_kv_heads=n_kv_heads,
                rope_base=rope_base,
                d_ffn=d_ffn,
                max_seq_len=max_seq_len,
                window_size=None if (i + 1) % global_attn_ratio == 0 else window_size,
                n_experts=n_experts,
                d_expert=d_expert,
                n_active=n_active,
            )
            self.layers.append(layer)

    def packed_masks(self, doc_ids: Tensor) -> dict[int | None, BlockMask | Tensor]:
        # One packed-document mask per distinct window size among the layers
        # (see _packed_attention_mask). Called by the training modules once
        # per step, *outside* the compiled forward: built inside it, the
        # graph break around the dynamo-disabled builder would drop the
        # layers out of torch.compile and run flex_attention unfused.
        masks: dict[int | None, BlockMask | Tensor] = {}
        for layer in self.layers:
            ws = layer.attn.window_size
            if ws not in masks:
                masks[ws] = _packed_attention_mask(doc_ids, ws)
        return masks

    def forward(
        self,
        x: Tensor,
        doc_ids: Tensor | None = None,
        masks: dict[int | None, BlockMask | Tensor] | None = None,
    ) -> Tensor:
        # doc_ids (b, l): id of the packed document each token belongs to.
        # When given, attention is confined within documents (sequence
        # packing). `masks` is the precomputed packed_masks(doc_ids) -- pass
        # it when this forward runs under torch.compile (see packed_masks);
        # otherwise it is built here for eager/CPU convenience.
        if masks is None and doc_ids is not None:
            masks = self.packed_masks(doc_ids)
        for layer in self.layers:
            mask = masks[layer.attn.window_size] if masks is not None else None
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, mask, use_reentrant=False
                )
            else:
                x = layer(x, mask)

        if self.training:
            # Apply each MoE's staged load-balancing bias update here -- once,
            # outside the checkpointed layer forwards above. Done inside
            # MoE.forward it would run twice under gradient checkpointing (see
            # MixtureOfExperts.forward).
            for layer in self.layers:
                if hasattr(layer, "moe"):
                    layer.moe.apply_bias_update()
        x = rms_norm(x)
        return x

    def decode(
        self, x: Tensor, cache: list[Tensor | None] | None = None, pos: int = 0
    ) -> tuple[Tensor, list[Tensor], int]:
        # Owns the absolute-position bookkeeping in one place: every layer sees
        # the same `pos` (they all process the same chunk at the same time), and
        # only this method computes/advances it. Neither the cache nor the
        # position is kept as model state -- both flow through args/returns only.
        if cache is None:
            cache = [None] * self.n_layers
        q_len = x.shape[-2]
        for i, layer in enumerate(self.layers):
            x, cache[i] = layer.decode(x, cache[i], pos)
        x = rms_norm(x)
        return x, cache, pos + q_len  # type: ignore


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_kv_heads: int | None = None,
        rope_base: int = 10000,
        d_ffn: int | None = None,
        max_seq_len: int = 4096,
        init_std: float = 0.02,
        grad_checkpoint: bool = True,
        window_size: int = 64,
        global_attn_ratio: int = 4,
        n_experts: int | None = None,
        n_active: int = 2,
        d_expert: int | None = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.init_std = init_std

        self.embed = nn.Embedding(vocab_size, d_model)
        self.lmhead = nn.Linear(d_model, vocab_size, bias=False)
        self.transformer = Transformer(
            d_model,
            n_heads,
            n_layers,
            n_kv_heads=n_kv_heads,
            rope_base=rope_base,
            d_ffn=d_ffn,
            max_seq_len=max_seq_len,
            grad_checkpoint=grad_checkpoint,
            window_size=window_size,
            global_attn_ratio=global_attn_ratio,
            n_experts=n_experts,
            n_active=n_active,
            d_expert=d_expert,
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
            if isinstance(m, SelfAttention):
                nn.init.normal_(m.proj_o.weight, mean=0.0, std=scaled_std)
            elif isinstance(m, SwiGLU):
                nn.init.normal_(m.proj_down.weight, mean=0.0, std=scaled_std)
            elif isinstance(m, MixtureOfExperts):
                nn.init.normal_(m.weight_down, mean=0.0, std=scaled_std)

    def encode(
        self,
        x: Tensor,
        doc_ids: Tensor | None = None,
        masks: dict[int | None, BlockMask | Tensor] | None = None,
    ) -> Tensor:
        # Shared trunk (embed + transformer) up to the final hidden state, before
        # the lm head. doc_ids/masks: see Transformer.forward (sequence packing).
        return self.transformer(self.embed(x), doc_ids, masks)

    def forward(
        self,
        x: Tensor,
        doc_ids: Tensor | None = None,
        masks: dict[int | None, BlockMask | Tensor] | None = None,
    ) -> Tensor:
        return self.lmhead(self.encode(x, doc_ids, masks))

    def decode(
        self, x: Tensor, cache: list[Tensor | None] | None = None, pos: int = 0
    ) -> tuple[Tensor, list[Tensor], int]:
        x = self.embed(x)
        x, cache, pos = self.transformer.decode(x, cache, pos)
        return self.lmhead(x), cache, pos


def estimate_num_params(
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_kv_heads: int | None = None,
    d_ffn: int | None = None,
    n_experts: int | None = None,
    d_expert: int | None = None,
    n_active: int = 2,
    active_only: bool = False,
    **_ignored,
) -> int:
    """Estimate a TransformerLM's parameter count from its hyperparameters,
    without building the model (which can OOM at large scale).

    active_only=False (default) counts every parameter (the total). active_only=
    True counts only the parameters that a single token's forward pass touches --
    the "active parameters" headline for a sparse model: the router still runs
    fully but only n_active of the n_experts experts fire per token. The
    embedding and the lm head are counted in full in both. (For a dense model
    the two are equal.)

    Mirrors the module shapes in this file. RMSNorm/RoPE add no parameters and
    buffers (e.g. MoE expert_bias) are ignored, so this is the trainable
    parameter count -- exact for the current architecture, but treat it as an
    estimate since the shapes may drift. Extra keyword arguments (max_seq_len,
    window_size, rope_base, ...) are accepted and ignored, so a preset or a
    saved model_config can be splatted straight in:

        estimate_num_params(**MODEL_PRESETS["base"])
        estimate_num_params(**MODEL_PRESETS["base"], active_only=True)
    """
    n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
    d_head = d_model // n_heads
    kv_dim = d_head * n_kv_heads
    ffn_hidden = 3 * d_model if d_ffn is None else d_ffn
    # TransformerLayer falls d_expert back to d_ffn, and MoE falls a None hidden
    # back to 3*d_model -- so an unset d_expert lands on ffn_hidden.
    expert_hidden = ffn_hidden if d_expert is None else d_expert

    # Attention: q/g/o are square (d_head*n_heads == d_model), k/v project to
    # kv_dim. All bias-free. g is the gated-attention output gate.
    attn = 3 * d_model * d_model + 2 * d_model * kv_dim
    # SwiGLU: up/gate/down, all bias-free.
    ffn = 3 * d_model * ffn_hidden
    layer_params = attn + ffn
    if n_experts is not None:
        # router (always fully active) + per-expert up/gate/down; only n_active
        # of the experts fire for a given token, so the active count uses those.
        experts = n_active if active_only else n_experts
        layer_params += n_experts * d_model + 3 * experts * expert_hidden * d_model

    embed = vocab_size * d_model
    lmhead = vocab_size * d_model  # separate (untied) output projection
    layers = n_layers * layer_params
    return embed + lmhead + layers


# Scale ladder: pico (~0.5B, the entry point) up to large (~23B MoE), kept in
# configs/presets.yml so the hyperparameters live with the other recipes (see
# that file for the sizing rationale).
PRESETS_FILE = Path(__file__).resolve().parents[1] / "configs" / "presets.yml"


def load_presets(path: str | Path = PRESETS_FILE) -> dict[str, dict]:
    """Load the {size: hyperparameters} scale ladder consumed by build_lm."""
    with open(path) as f:
        return yaml.safe_load(f)


MODEL_PRESETS: dict[str, dict] = load_presets()


def build_lm(
    size: str,
    vocab_size: int | None = None,
    max_seq_len: int = 4096,
    **overrides,
) -> TransformerLM:
    """Build a TransformerLM from a preset name. vocab_size defaults to the
    preset's recommended value; pass it explicitly (e.g. the tokenizer's actual
    vocab) to override. Any other field can be overridden via overrides."""
    if size not in MODEL_PRESETS:
        raise ValueError(f"unknown size '{size}'. choices: {list(MODEL_PRESETS)}")
    cfg = {**MODEL_PRESETS[size], **overrides}
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return TransformerLM(max_seq_len=max_seq_len, **cfg)


def estimate_preset_params(
    size: str,
    vocab_size: int | None = None,
    active_only: bool = False,
    **overrides,
) -> int:
    """Estimate the parameter count of build_lm(size, ...) without building it.

    Same preset/override resolution as build_lm, so the two always describe the
    same model. Handy for sizing the scale ladder on a machine that can't hold
    the larger presets in memory. active_only=True returns the per-token active
    parameter count instead of the total (see estimate_num_params)."""
    if size not in MODEL_PRESETS:
        raise ValueError(f"unknown size '{size}'. choices: {list(MODEL_PRESETS)}")
    cfg = {**MODEL_PRESETS[size], **overrides}
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return estimate_num_params(**cfg, active_only=active_only)
