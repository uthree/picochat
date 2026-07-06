import math

import lightning as L
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

from picochat.optim import Muon


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
        # gradient, no loss term, just a running buffer updated in-place at
        # the end of forward.
        self.register_buffer("expert_bias", torch.zeros(n_experts))

    def forward(self, x: Tensor) -> Tensor:
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
        capacity = max(
            self.n_active,
            math.ceil(n_tokens * self.n_active * self.capacity_factor / self.n_experts),
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
            with torch.no_grad():
                load = filled.float()
                self.expert_bias += self.bias_update_rate * torch.sign(
                    load.mean() - load
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
        n_lmheads: int = 1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.init_std = init_std

        self.embed = nn.Embedding(vocab_size, d_model)
        self.lmheads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size, bias=False) for _ in range(n_lmheads)]
        )
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
        # GPT-2 style: init every weight with normal(0, init_std) and zero biases,
        # then scale down the projections that write into the residual stream
        # (proj_o / proj_down) by 1/sqrt(2*n_layers) so the residual variance stays
        # roughly constant with depth.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
        scaled_std = self.init_std / math.sqrt(2 * self.n_layers)
        for m in self.modules():
            if isinstance(m, SelfAttention):
                nn.init.normal_(m.proj_o.weight, mean=0.0, std=scaled_std)
            elif isinstance(m, SwiGLU):
                nn.init.normal_(m.proj_down.weight, mean=0.0, std=scaled_std)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.embed(x)
        x = self.transformer(x)
        return [h_t(x) for h_t in self.lmheads]

    def decode(
        self, x: Tensor, cache: list[Tensor | None] | None = None, pos: int = 0
    ) -> tuple[Tensor, list[Tensor], int]:
        x = self.embed(x)
        x, cache, pos = self.transformer.decode(x, cache, pos)
        # Autoregressive decoding only consumes the next-token head; the extra
        # MTP heads are a training-time signal (and a future hook for
        # self-speculative decoding).
        return self.lmheads[0](x), cache, pos


# Scale ladder.
MODEL_PRESETS: dict[str, dict] = {
    "pico": dict(
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=64000,
        window_size=128,
        global_attn_ratio=4,
        n_experts=8,
        d_expert=1024,
        n_lmheads=4,
    ),
    "small": dict(
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        vocab_size=64000,
        window_size=128,
        global_attn_ratio=4,
        n_experts=16,
        d_expert=1024,
        n_lmheads=4,
    ),
    "base": dict(
        d_model=1024,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=64000,
        window_size=128,
        global_attn_ratio=6,
        n_experts=32,
        d_expert=1024,
        n_lmheads=4,
    ),
    "medium": dict(
        d_model=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=64000,
        window_size=256,
        global_attn_ratio=6,
        n_experts=64,
        d_expert=1024,
        n_lmheads=4,
    ),
    "large": dict(
        d_model=2560,
        n_layers=32,
        n_heads=20,
        n_kv_heads=5,
        vocab_size=64000,
        window_size=256,
        global_attn_ratio=6,
        n_experts=128,
        d_expert=1024,
        n_lmheads=4,
    ),
}


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


def can_compile() -> bool:
    """Whether torch.compile is likely to help in this environment.

    The inductor backend targets CUDA; on CPU/MPS it often falls back or errors,
    so we only enable it on CUDA. torch.compile itself is lazy (compiles on the
    first forward), so this just gates whether we wrap the model at all.
    """
    return hasattr(torch, "compile") and torch.cuda.is_available()


class GPT(L.LightningModule):
    def __init__(
        self,
        transformer_lm: TransformerLM,
        pad_idx: int = 0,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        optimizer: str = "muon",
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        warmup_steps: int = 2000,
        max_steps: int | None = None,
        min_lr_ratio: float = 0.1,
        compile: bool | None = None,
        tokenizer=None,
        sample_batches: int = 20,
        model_config: dict | None = None,
    ):
        super().__init__()
        # `model_config` is the plain-dict build_lm(**model_config) recipe used to
        # construct `transformer_lm` (size/vocab_size/max_seq_len/overrides).
        # Saving it (and nothing else -- transformer_lm/tokenizer aren't
        # cleanly picklable/yaml-able) lets a checkpoint's own
        # hyper_parameters rebuild the exact same architecture later, instead
        # of relying on the caller to pass matching flags by hand.
        self.save_hyperparameters("model_config")
        self.model = transformer_lm
        self.pad_idx = pad_idx
        # Optional tiktoken Encoding used to turn generated token ids back into
        # readable text for the TensorBoard generation samples (see below).
        self.tokenizer = tokenizer
        # During validation, log a generated continuation for batches with
        # batch_idx <= sample_batches (decode is slow, so only the first few).
        self.sample_batches = sample_batches
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        # "muon" (default) or "adamw". With muon, `lr`/`betas` still apply --
        # to the embedded AdamW that handles the params Muon skips.
        self.optimizer_name = optimizer
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        # `compile=None` -> auto (compile iff the environment supports it). The
        # compiled handle shares parameters with self.model; we stash it inside a
        # list so nn.Module doesn't register it as a submodule (which would
        # duplicate every parameter under a `_train_model._orig_mod.` prefix and
        # break checkpoint loading). self.model stays uncompiled, so state_dict
        # keys stay clean and decode() runs eager.
        self.compile = can_compile() if compile is None else compile
        self._train_model = [torch.compile(self.model) if self.compile else self.model]

    def _head_losses(self, x: Tensor) -> Tensor:
        # Multiple token prediction: head k's output at position i predicts
        # token i+1+k, so each head shifts the targets one step further (head 0
        # is ordinary next-token prediction). Returns one loss per head.
        head_logits = self._train_model[0](x)
        losses = []
        for k, logits in enumerate(head_logits):
            shift = k + 1
            logits = rearrange(logits[:, :-shift], "b l v -> (b l) v")
            targets = rearrange(x[:, shift:], "b l -> (b l)")
            losses.append(F.cross_entropy(logits, targets, ignore_index=self.pad_idx))
        return torch.stack(losses)

    def _loss(self, x: Tensor) -> Tensor:
        return self._head_losses(x).mean()

    def _log_head_losses(self, prefix: str, head_losses: Tensor) -> None:
        # Per-head breakdown (head 0 is the loss comparable to a single-head
        # run); skip when there's nothing to break down.
        if head_losses.numel() > 1:
            for k, head_loss in enumerate(head_losses):
                self.log(f"{prefix}_head{k}", head_loss)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        head_losses = self._head_losses(batch)
        loss = head_losses.mean()
        self.log("train_loss", loss)
        self._log_head_losses("train_loss", head_losses)
        self.log("loss", loss, prog_bar=True, logger=False)  # for progress bar
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        head_losses = self._head_losses(batch)
        loss = head_losses.mean()
        self.log("val_loss", loss, prog_bar=True)
        self._log_head_losses("val_loss", head_losses)
        if batch_idx <= self.sample_batches:
            # Sanity-check what the model actually generates: prefill the first
            # half of the sequence and let it autoregress the second half, then
            # log prompt/generated/reference side by side to TensorBoard.
            self._log_generation_sample(batch, batch_idx)
        return loss

    @torch.no_grad()
    def _generate(self, prompt: Tensor, max_new_tokens: int) -> Tensor:
        """Greedy-decode `max_new_tokens` tokens after `prompt` (B, L) via KV cache."""
        # `pos` tracks the absolute decode position as a plain local int -- not
        # model state -- and is threaded through each call, same as `cache`.
        logits, cache, pos = self.model.decode(prompt)
        next_token = logits[:, -1:].argmax(dim=-1)
        out = [next_token]
        for _ in range(max_new_tokens - 1):
            logits, cache, pos = self.model.decode(next_token, cache, pos)
            next_token = logits[:, -1:].argmax(dim=-1)
            out.append(next_token)
        return torch.cat(out, dim=1)  # (B, max_new_tokens)

    def _decode_text(self, ids: Tensor) -> str:
        try:
            return self.tokenizer.decode(ids.tolist())
        except Exception:
            return "<decode error>"

    def _log_generation_sample(self, batch: Tensor, batch_idx: int) -> None:
        # Need a tokenizer to render text and a TensorBoard writer to log it.
        writer = getattr(self.logger, "experiment", None)
        if self.tokenizer is None or writer is None or not hasattr(writer, "add_text"):
            return
        seq = batch[0]  # one example per logged batch is enough
        half = seq.shape[0] // 2
        if half == 0:
            return
        prompt, reference = seq[:half], seq[half:]
        generated = self._generate(prompt[None], max_new_tokens=reference.shape[0])[0]
        text = (
            f"**prompt**\n\n{self._decode_text(prompt)}\n\n"
            f"**generated**\n\n{self._decode_text(generated)}\n\n"
            f"**reference**\n\n{self._decode_text(reference)}"
        )
        writer.add_text(f"val_sample/{batch_idx}", text, self.global_step)

    def _param_groups(self) -> list[dict]:
        # Apply weight decay only to weights with 2+ dims. Exclude biases (1-dim)
        # and embeddings (rms_norm has no learnable params, so nothing to exclude there).
        embed_ids = {
            id(p)
            for m in self.model.modules()
            if isinstance(m, nn.Embedding)
            for p in m.parameters()
        }
        decay, no_decay = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or id(p) in embed_ids:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _lr_lambda(self, step: int) -> float:
        # Linear warmup -> cosine decay (down to min_lr_ratio).
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        if self.max_steps is None or step >= self.max_steps:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * coeff

    def _muon_param_groups(self) -> list[dict]:
        # Muon orthogonalizes matrix-shaped *hidden* weights. The embedding and
        # lm heads (input/output layers, per the Muon authors) and 1-dim params
        # (biases) go to the embedded AdamW instead, keeping the same decay
        # split as _param_groups: no decay for embeddings/1-dim, decay for the
        # lm-head matrices. Everything else -- attention/FFN projections, the
        # router, and the fused 3D MoE expert weights (flattened inside Muon.step)
        # -- is optimized by Muon.
        embed_ids = {
            id(p)
            for m in self.model.modules()
            if isinstance(m, nn.Embedding)
            for p in m.parameters()
        }
        head_ids = {
            id(p) for head in self.model.lmheads for p in head.parameters()
        }
        muon, adam_decay, adam_no_decay = [], [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or id(p) in embed_ids:
                adam_no_decay.append(p)
            elif id(p) in head_ids:
                adam_decay.append(p)
            else:
                muon.append(p)
        return [
            dict(
                params=muon,
                use_muon=True,
                lr=self.muon_lr,
                momentum=self.muon_momentum,
                weight_decay=self.weight_decay,
            ),
            dict(
                params=adam_decay,
                use_muon=False,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            ),
            dict(
                params=adam_no_decay,
                use_muon=False,
                lr=self.lr,
                betas=self.betas,
                weight_decay=0.0,
            ),
        ]

    def configure_optimizers(self):
        if self.optimizer_name == "muon":
            optimizer = Muon(self._muon_param_groups())
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self._param_groups(), lr=self.lr, betas=self.betas
            )
        else:
            raise ValueError(
                f"unknown optimizer '{self.optimizer_name}'. choices: muon, adamw"
            )
        if self.max_steps is None:
            # No schedule when the training horizon is unknown (optimizer only).
            return optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
