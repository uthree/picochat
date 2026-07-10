"""Transformer language-model architecture: the building blocks (norm, RoPE,
SwiGLU, MoE, attention) up through TransformerLM. The GPT LightningModule that
trains it, plus the scale-ladder presets, live in gpt.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.proj_up = nn.Linear(d_model, d_hidden)
        self.proj_gate = nn.Linear(d_model, d_hidden)
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
        self.weight_router = nn.Parameter(torch.empty(n_experts, d_model))
        self.weight_up = nn.Parameter(torch.empty(n_experts, d_hidden, d_model))
        self.weight_gate = nn.Parameter(torch.empty(n_experts, d_hidden, d_model))
        self.weight_down = nn.Parameter(torch.empty(n_experts, d_model, d_hidden))
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
        # no_drop=True (used only by decode) never drops a token; the training /
        # validation forward keeps the bounded, fixed-capacity path. See the
        # capacity computation below.
        b, l, d = x.shape
        n_tokens = b * l
        tokens = rms_norm(x).reshape(n_tokens, d)

        # Route every token to its top-n_active experts. The bias only affects
        # *which* experts are selected; the combine weight is still softmax
        # over the real (unbiased) logits, so it stays a differentiable
        # function of weight_router alone.
        logits = tokens @ self.weight_router.T  # (n_tokens, n_experts)
        top_idx = (logits + self.expert_bias).topk(self.n_active, dim=-1).indices
        top_logits = logits.gather(-1, top_idx)
        top_weight = F.softmax(top_logits, dim=-1)  # (n_tokens, n_active)

        # Fixed per-expert capacity keeps every tensor below a static shape --
        # unlike a data-dependent gather/nonzero, this stays traceable under
        # torch.compile and shards cleanly across devices for future
        # expert-parallel training. Tokens beyond an expert's capacity are
        # dropped for that expert (Switch Transformer / GShard style); within
        # a token, slot 0 (its top choice) claims capacity before slot 1, etc.
        if no_drop:
            # Decode/generation: never drop a token. n_tokens is the worst case
            # (all tokens to one expert), so nothing overflows -- the layer
            # becomes purely per-token, so chunked decode matches a full no-drop
            # forward exactly and no generated token is silently dropped. Only
            # reached from decode(), where n_tokens is small, so the larger,
            # mostly-empty buffer is cheap. The training/validation forward keeps
            # the bounded path below: bounding it here too would blow the buffer
            # up n_experts/(n_active*capacity_factor)x on full-batch validation.
            capacity = n_tokens
        else:
            capacity = max(
                self.n_active,
                math.ceil(
                    n_tokens * self.n_active * self.capacity_factor / self.n_experts
                ),
            )
        n_slots = self.n_experts * capacity
        # Row n_slots is a trash bin: every dropped token's slot is redirected
        # there instead of a real expert, so index_add/index_select below can
        # use a fixed-size index (no dynamic-length gather) while still
        # discarding the overflow.
        buffer = tokens.new_zeros(n_slots + 1, d)
        filled = torch.zeros(self.n_experts, dtype=torch.long, device=x.device)
        dests, keeps = [], []
        for slot in range(self.n_active):
            expert_idx = top_idx[:, slot]  # (n_tokens,)
            one_hot = F.one_hot(expert_idx, self.n_experts)  # int64, exact counts
            position = one_hot.cumsum(dim=0) - 1 + filled
            token_position = position.gather(-1, expert_idx.unsqueeze(-1)).squeeze(-1)
            keep = token_position < capacity
            dest = torch.where(
                keep,
                expert_idx * capacity + token_position,
                torch.full_like(expert_idx, n_slots),
            )
            buffer = buffer.index_add(
                0, dest, tokens * keep.unsqueeze(-1).to(tokens.dtype)
            )
            filled = filled + one_hot.sum(dim=0)
            dests.append(dest)
            keeps.append(keep)

        if self.training:
            # Stage the bias delta instead of applying it here. Under gradient
            # checkpointing this forward is run twice (once for real, once
            # recomputed in backward), so an in-place `expert_bias +=` would
            # double the update -- and worse, the recompute would then route
            # against an already-shifted bias, diverging from the forward it is
            # meant to reproduce. Staging into a buffer is idempotent (same value
            # each run); Transformer.forward applies it exactly once, outside the
            # checkpoint. expert_bias itself stays constant across the pair.
            with torch.no_grad():
                load = filled.float()
                self._pending_bias_delta.copy_(
                    self.bias_update_rate * torch.sign(load.mean() - load)
                )

        expert_in = buffer[:n_slots].reshape(self.n_experts, capacity, d)
        up = torch.bmm(expert_in, self.weight_up.transpose(1, 2))
        gate = torch.bmm(expert_in, self.weight_gate.transpose(1, 2))
        h = F.dropout(up * F.silu(gate), self.p_dropout, training=self.training)
        expert_out = torch.bmm(h, self.weight_down.transpose(1, 2))
        expert_out = F.dropout(expert_out, self.p_dropout, training=self.training)
        expert_out = torch.cat(
            [expert_out.reshape(n_slots, d), expert_out.new_zeros(1, d)], dim=0
        )

        out = tokens.new_zeros(n_tokens, d)
        for slot in range(self.n_active):
            picked = expert_out.index_select(0, dests[slot])
            coeff = (top_weight[:, slot] * keeps[slot]).unsqueeze(-1).to(tokens.dtype)
            out = out + picked * coeff

        return out.reshape(b, l, d)

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
        self.proj_o = nn.Linear(self.d_head * n_heads, d_model, bias=False)

        sin, cos = self._rope_tables(max_seq_len)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Shared q/k/v projection (with QK-norm) for both forward and decode.
        x = rms_norm(x)
        query = rearrange(self.proj_q(x), "b l (h d) -> b h l d", d=self.d_head)
        key = rms_norm(rearrange(self.proj_k(x), "b l (g d) -> b g l d", d=self.d_head))
        value = rms_norm(
            rearrange(self.proj_v(x), "b l (g d) -> b g l d", d=self.d_head)
        )
        return query, key, value

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

    def forward(self, x: Tensor) -> Tensor:
        # Training path: full causal attention over the whole sequence, no cache.
        query, key, value = self._project(x)
        query, key = self._rope(query), self._rope(key)
        if self.window_size is None:
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
        return self.proj_o(rearrange(attn, "b h l d -> b l (h d)"))

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
        query, key, value = self._project(x)
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
        out = self.proj_o(rearrange(attn, "b h l d -> b l (h d)"))
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

    def forward(self, x: Tensor) -> Tensor:
        # attn/ffn apply pre-norm (rms_norm) internally, so add the raw residual.
        x = self.attn(x) + x
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

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

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
        init_std: float = 0.01,
        grad_checkpoint: bool = True,
        window_size: int = 64,
        global_attn_ratio: int = 4,
        n_experts: int | None = None,
        n_active: int = 2,
        d_expert: int | None = None,
        n_lmheads: int = 1,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.init_std = init_std
        self.tie_embeddings = tie_embeddings

        self.embed = nn.Embedding(vocab_size, d_model)
        self.lmheads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size, bias=False) for _ in range(n_lmheads)]
        )
        if tie_embeddings:
            # Share the next-token head's weight with the embedding (the extra
            # MTP heads, if any, stay untied). Assigning after construction
            # drops the head's own Parameter and replaces it with the same
            # tensor object as self.embed.weight, so the two stay in sync
            # through training/optimization/checkpointing for free.
            self.lmheads[0].weight = self.embed.weight
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
        # Every weight is init'd from the same normal(0, init_std); biases are
        # zeroed. init_std is small enough (well under GPT-2's 0.02) that this
        # needs no per-layer/depth scaling to stay stable, unlike GPT-2 style
        # init -- and staying simple/uniform matters here since checkpoints get
        # loaded into larger configs via load_state_dict_expand, where the
        # untouched (randomly initialized) region should look statistically
        # identical to every other weight, not depend on which layer it's in.
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: Tensor) -> Tensor:
        # Shared trunk (embed + transformer) up to the final hidden state, before
        # the lm heads. Split out so training can backprop each MTP head
        # separately off one trunk forward (see GPT._mtp_backward).
        return self.transformer(self.embed(x))

    def forward(self, x: Tensor) -> list[Tensor]:
        h = self.encode(x)
        return [h_t(h) for h_t in self.lmheads]

    def decode(
        self, x: Tensor, cache: list[Tensor | None] | None = None, pos: int = 0
    ) -> tuple[Tensor, list[Tensor], int]:
        x = self.embed(x)
        x, cache, pos = self.transformer.decode(x, cache, pos)
        # Autoregressive decoding only consumes the next-token head; the extra
        # MTP heads are a training-time signal (and a future hook for
        # self-speculative decoding).
        return self.lmheads[0](x), cache, pos


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
    n_lmheads: int = 1,
    active_only: bool = False,
    tie_embeddings: bool = False,
    **_ignored,
) -> int:
    """Estimate a TransformerLM's parameter count from its hyperparameters,
    without building the model (which can OOM at large scale).

    active_only=False (default) counts every parameter (the total). active_only=
    True counts only the parameters that a single token's forward pass touches --
    the "active parameters" headline for a sparse model: the router still runs
    fully but only n_active of the n_experts experts fire per token, and only the
    next-token head (lmheads[0]) is used, so the extra MTP heads drop out. The
    embedding is counted in full in both. (For a dense model the two are equal.)

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

    # Attention: q/o are square (d_head*n_heads == d_model), k/v project to
    # kv_dim. All bias-free.
    attn = 2 * d_model * d_model + 2 * d_model * kv_dim
    # SwiGLU: up/gate carry a bias, down is bias-free.
    ffn = 3 * d_model * ffn_hidden + 2 * ffn_hidden
    layer_params = attn + ffn
    if n_experts is not None:
        # router (always fully active) + per-expert up/gate/down; only n_active
        # of the experts fire for a given token, so the active count uses those.
        experts = n_active if active_only else n_experts
        layer_params += n_experts * d_model + 3 * experts * expert_hidden * d_model

    embed = vocab_size * d_model
    # heads are untied by default, one Linear each; only lmheads[0] runs
    # autoregressively. With tie_embeddings, lmheads[0] shares embed's weight
    # (already counted in `embed`), so only the extra MTP heads (n_lmheads-1)
    # add parameters, and active_only (which only touches lmheads[0]) adds none.
    if tie_embeddings:
        n_heads_counted = 0 if active_only else max(0, n_lmheads - 1)
    else:
        n_heads_counted = 1 if active_only else n_lmheads
    lmheads = n_heads_counted * vocab_size * d_model
    layers = n_layers * layer_params
    return embed + lmheads + layers
