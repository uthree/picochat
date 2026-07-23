"""Gated DeltaNet-2: the linear-attention (recurrent) mixer that replaces
sliding-window softmax attention in picochat's hybrid stack.

Gated DeltaNet-2 (arXiv:2605.22791, "GDN-2") refines Gated DeltaNet /
KDA-style delta-rule models by decoupling the single scalar gate that used to
control both memory erasure and memory commit into two independent
channel-wise gates: an erase gate b_t on the key axis and a write gate w_t on
the value axis, on top of KDA's channel-wise log-decay g_t. Per token, on the
matrix state S in R^{K x V} (one per head):

    S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T
    o_t = S_t^T q_t

with `*` the elementwise product; collapsing b_t = w_t = beta to a scalar
recovers KDA / Gated DeltaNet. The recurrence is inherently order-dependent,
so these layers carry position information implicitly and use *no* RoPE (the
softmax layers -- picochat's NSA layers -- keep their own positional scheme).
Two equivalent evaluation paths:

- training: a chunkwise-parallel form (`chunk_gdn2`) that is matmul-heavy,
  expanded via the gated WY representation (mirrors fla's reference);
- decode: the plain O(1)-per-token recurrence (`recurrent_gdn2`).

Both are exact pure-PyTorch and run on CPU (the correctness oracle /
fallback). On CUDA the `fla` (flash-linear-attention) Triton kernels are used
when installed. Sequence packing resets the recurrent state at document
boundaries via `cu_seqlens` (segment-wise here; native in fla).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:  # CUDA Triton acceleration (fla-core, a Linux dependency); guarded so
    # macOS/Windows -- where triton isn't installed -- fall back cleanly, and
    # everything still works CPU-side without it (see the pure-PyTorch kernels
    # below, used whenever fla is absent or the tensors aren't on CUDA).
    from fla.ops.gdn2 import (  # type: ignore
        chunk_gdn2 as _fla_chunk_gdn2,
        fused_recurrent_gdn2 as _fla_recurrent_gdn2,
    )

    _HAS_FLA = True
except Exception:  # pragma: no cover - depends on the environment
    _HAS_FLA = False


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """L2-normalize along `dim` (the feature-map normalization on q/k; matches
    fla's kernel-side l2norm)."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def recurrent_gdn2(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    initial_state: Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[Tensor, Tensor]:
    """Sequential GDN-2 rule (the O(1)-per-token decode path).

    Shapes: query/key (B, T, H, Dk), value (B, T, H, Dv), g/b (B, T, H, Dk),
    w (B, T, H, Dv), with g the channel-wise *log* decay (<= 0), b the
    channel-wise erase gate and w the channel-wise write gate (both typically
    in (0, 1)). Returns the outputs (B, T, H, Dv) and the final state
    (B, H, Dk, Dv). Runs in float32 for stability. Mirrors fla's
    naive_recurrent_gdn2 (plus the q/k feature-map normalization).
    """
    dtype = query.dtype
    if use_qk_l2norm:
        query, key = l2norm(query), l2norm(key)
    query, key, value, g, b, w = (
        t.transpose(1, 2).contiguous().float() for t in (query, key, value, g, b, w)
    )
    bsz, h, t, dk = key.shape
    dv = value.shape[-1]
    query = query * dk**-0.5

    out = torch.zeros(bsz, h, t, dv, dtype=torch.float32, device=value.device)
    state = (
        torch.zeros(bsz, h, dk, dv, dtype=torch.float32, device=value.device)
        if initial_state is None
        else initial_state.float()
    )
    for i in range(t):
        k_t = key[:, :, i]  # (bsz, h, dk)
        v_t = value[:, :, i]  # (bsz, h, dv)
        state = state * g[:, :, i].exp()[..., None]  # channel-wise decay on K
        erase = ((b[:, :, i] * k_t)[..., None] * state).sum(-2)  # (b*k)^T S
        delta = w[:, :, i] * v_t - erase  # gated write minus gated read
        state = state + k_t[..., None] * delta[..., None, :]  # + k (.)^T
        out[:, :, i] = (state * query[:, :, i][..., None]).sum(-2)  # q^T S
    return out.transpose(1, 2).contiguous().to(dtype), state


def chunk_gdn2(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    initial_state: Tensor | None = None,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
) -> tuple[Tensor, Tensor]:
    """Chunkwise-parallel GDN-2 rule (the training path): exactly equal to the
    sequential recurrence but expressed as batched matmuls over length-C
    chunks. The within-chunk recurrence is expanded via the WY representation
    `A = (I + tril((b*k*exp(g)) k^T, -1))^{-1}`, then `u = A (w * v)` and
    `w_wy = A (b * k * exp(g))`; the inter-chunk state recurrence carries the
    update across chunks. Same shapes / return contract as recurrent_gdn2.
    Mirrors fla's naive_chunk_gdn2.
    """
    dtype = query.dtype
    if use_qk_l2norm:
        query, key = l2norm(query), l2norm(key)
    query, key, value, g, b, w = (
        t.transpose(1, 2).contiguous().float() for t in (query, key, value, g, b, w)
    )
    bsz, h, t, dk = key.shape
    dv = value.shape[-1]
    # The chunk size only sets the parallelization granularity, not the result
    # (exactly equal to the sequential recurrence for any value). Cap it at the
    # sequence length so short sequences don't pay the full-chunk Python loops.
    c = max(1, min(chunk_size, t))
    pad = (c - t % c) % c
    query, key, value, g, b, w = (
        F.pad(x, (0, 0, 0, pad)) for x in (query, key, value, g, b, w)
    )
    tt = t + pad
    nt = tt // c
    query = query * dk**-0.5

    def chunk(x: Tensor) -> Tensor:
        return x.reshape(bsz, h, nt, c, x.shape[-1])

    query, key, value, g, b, w = (chunk(x) for x in (query, key, value, g, b, w))

    g_cum = g.cumsum(-2)  # (bsz, h, nt, c, dk): within-chunk cumulative log decay
    g_last = g_cum[..., -1:, :]  # decay to the chunk end (the state carry factor)
    tril = torch.tril(torch.ones(c, c, dtype=torch.bool, device=query.device), -1)
    causal = torch.tril(torch.ones(c, c, dtype=torch.bool, device=query.device), 0)
    # Pairwise decay factors exp(g_cum[i] - g_cum[j]), per channel. Only the
    # causal (i >= j) entries are ever used; the anticausal ones have *positive*
    # exponents that overflow to inf for strong decays -- masking them after the
    # exp would still poison the backward (0 * inf = nan), so zero the exponent
    # first (their downstream products are masked out either way).
    decay_ij = (
        (g_cum.unsqueeze(-2) - g_cum.unsqueeze(-3))
        .masked_fill(~causal[:, :, None], 0.0)
        .exp()
    )  # (..., c, c, dk)

    # (I + tril((b*k) K^T decay, -1))^{-1}, built by forward substitution.
    bk = b * key
    t_lower = torch.einsum("bhnik,bhnjk,bhnijk->bhnij", bk, key, decay_ij)
    t_lower = t_lower.masked_fill(~tril, 0.0)
    attn = -t_lower
    for i in range(1, c):
        row = attn[..., i, :i].clone()
        attn[..., i, :i] = row + (row[..., None] * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(c, dtype=attn.dtype, device=attn.device)

    u_wy = attn @ (w * value)  # pseudo-values U
    w_wy = attn @ (bk * g_cum.exp())  # A (b * k * exp(g))
    k_tail = key * (g_last - g_cum).exp()  # key decayed to the chunk end

    state = (
        torch.zeros(bsz, h, dk, dv, dtype=torch.float32, device=value.device)
        if initial_state is None
        else initial_state.float()
    )
    out = torch.zeros(bsz, h, nt, c, dv, dtype=torch.float32, device=value.device)
    for i in range(nt):
        q_i, k_i = query[:, :, i], key[:, :, i]
        g_i, g_last_i = g_cum[:, :, i], g_last[:, :, i].squeeze(-2)
        v_new = u_wy[:, :, i] - w_wy[:, :, i] @ state  # delta write minus carry
        a = torch.einsum(
            "bhik,bhjk,bhijk->bhij", q_i, k_i, decay_ij[:, :, i]
        ).masked_fill(~causal, 0.0)
        out[:, :, i] = a @ v_new + (q_i * g_i.exp()) @ state
        state = (
            state * g_last_i.unsqueeze(-1).exp()
            + k_tail[:, :, i].transpose(-1, -2) @ v_new
        )

    out = out.reshape(bsz, h, tt, dv)[:, :, :t]
    return out.transpose(1, 2).contiguous().to(dtype), state


def _segmented_chunk(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    cu_seqlens: Tensor,
    chunk_size: int,
) -> Tensor:
    """Run the chunk rule independently on each `cu_seqlens` segment so the
    recurrent state resets at every document boundary (sequence packing). Inputs
    are flattened to a single row (B=1, T=total); returns (1, T, H, Dv). Used by
    the pure-PyTorch path; fla accepts cu_seqlens natively."""
    outs = []
    for s, e in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        o, _ = chunk_gdn2(
            query[:, s:e],
            key[:, s:e],
            value[:, s:e],
            g[:, s:e],
            b[:, s:e],
            w[:, s:e],
            chunk_size=chunk_size,
        )
        outs.append(o)
    return torch.cat(outs, dim=1)


class GatedRMSNorm(nn.Module):
    """RMSNorm with a SiLU output gate: weight * rms_norm(x) * silu(gate). The
    per-head output normalization + gate at the end of a Gated DeltaNet-2 mixer
    (matches Qwen3-Next's RMSNormGated)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(dtype)
        return (x * F.silu(gate.float())).to(dtype)


class GatedDeltaNet2(nn.Module):
    """Gated DeltaNet-2 mixer (drop-in replacement for a sliding-window
    SelfAttention layer). Mirrors picochat's GQA head convention: `n_heads`
    value/output heads and `n_kv_heads` key heads (n_heads a multiple of
    n_kv_heads; q/k -- and the key-side gates g/b -- are repeat-interleaved up
    to n_heads for the recurrence), with a square head dim `d_model // n_heads`
    (so the value dim is d_model and out_proj is square). No RoPE -- positions
    come from the recurrence itself.

    Gate parameterization (following the reference fla GDN-2 layer):
      * g -- channel-wise log decay on the key axis, Mamba2-style
        `g = -exp(A_log) * softplus(f_proj(x) + dt_bias)` with A_log per key
        head and dt_bias per key channel; f_proj is a low-rank (bottleneck
        d_head) projection so the channel-wise gate stays cheap.
      * b -- channel-wise erase gate on the key axis, sigmoid.
      * w -- channel-wise write gate on the value axis, sigmoid.

    forward(x, cu_seqlens) is the packed training path; decode(x, state) is the
    O(1)-per-token inference step carrying (recurrent_state, conv_state).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        conv_size: int = 4,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads  # value/output heads (num_v_heads)
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert n_heads % self.n_kv_heads == 0, (
            "n_heads must be a multiple of n_kv_heads"
        )
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "heads must tile d_model"
        self.n_rep = n_heads // self.n_kv_heads
        self.conv_size = conv_size
        self.chunk_size = chunk_size

        self.key_dim = self.n_kv_heads * self.d_head
        self.value_dim = self.n_heads * self.d_head  # == d_model
        self.conv_dim = 2 * self.key_dim + self.value_dim

        self.proj_q = nn.Linear(d_model, self.key_dim, bias=False)
        self.proj_k = nn.Linear(d_model, self.key_dim, bias=False)
        self.proj_v = nn.Linear(d_model, self.value_dim, bias=False)
        self.proj_z = nn.Linear(d_model, self.value_dim, bias=False)  # output gate
        # GDN-2's decoupled channel-wise gates. f feeds the log decay (low-rank,
        # bottleneck d_head), b the erase strength (key axis), w the write
        # strength (value axis).
        self.f_proj = nn.Sequential(
            nn.Linear(d_model, self.d_head, bias=False),
            nn.Linear(self.d_head, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.w_proj = nn.Linear(d_model, self.value_dim, bias=False)
        # Mamba2 gate parameterization: g = -exp(A_log) * softplus(f + dt_bias),
        # A_log per key head, dt_bias per key channel.
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim))
        self.A_log = nn.Parameter(torch.zeros(self.n_kv_heads))
        # Depthwise causal short conv over concat(q, k, v) before the recurrence.
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=conv_size,
            groups=self.conv_dim,
            padding=conv_size - 1,
            bias=False,
        )
        self.norm = GatedRMSNorm(self.d_head)
        self.proj_o = nn.Linear(self.value_dim, d_model, bias=False)

    def reset_parameters(self) -> None:
        # Called from TransformerLM._init_weights after the GPT-2 normal init, to
        # restore the gate parameters' intended (non-normal) initialization.
        with torch.no_grad():
            self.A_log.copy_(torch.log(torch.empty(self.n_kv_heads).uniform_(1, 16)))
            self.dt_bias.zero_()
            self.norm.weight.fill_(1.0)

    def _gates(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Returns (g, b, w) in head layout: g/b (bsz, seq, n_heads, d_head) on
        # the key axis (computed per key head, repeated with q/k), w
        # (bsz, seq, n_heads, d_head) on the value axis.
        bsz, seq = x.shape[:2]
        f = self.f_proj(x).float() + self.dt_bias
        g = -self.A_log.float().exp()[:, None] * F.softplus(
            f.reshape(bsz, seq, self.n_kv_heads, self.d_head)
        )
        b = self.b_proj(x).sigmoid().reshape(bsz, seq, self.n_kv_heads, self.d_head)
        if self.n_rep > 1:
            g = g.repeat_interleave(self.n_rep, dim=2)
            b = b.repeat_interleave(self.n_rep, dim=2)
        w = self.w_proj(x).sigmoid().reshape(bsz, seq, self.n_heads, self.d_head)
        return g, b, w

    def _conv(self, qkv: Tensor, doc_ids: Tensor | None = None) -> Tensor:
        # qkv: (b, l, conv_dim). Causal depthwise conv + SiLU, applied per row
        # (left-pad by conv_size-1 then trim). Decode handles its own conv state
        # in decode(); this is the training/prefill path.
        #
        # With `doc_ids` (sequence packing), the conv must not read across a
        # document boundary -- otherwise a doc's first few tokens mix in the
        # previous doc's tail, and stacked GDN layers compound that leak past the
        # boundary. We zero every kernel tap whose source token is in a different
        # document than the output token (a vectorized, torch.compile-friendly
        # unfold, equivalent to fla's causal_conv1d seq_idx reset). With a single
        # document this is exactly the plain causal conv.
        x = qkv.transpose(1, 2)  # (b, conv_dim, l)
        if doc_ids is None:
            y = self.conv1d(x)[..., : x.shape[-1]]
            return F.silu(y).transpose(1, 2)
        b, c, seq = x.shape
        k = self.conv_size
        xp = F.pad(x, (k - 1, 0))  # (b, c, seq + k - 1)
        win = xp.unfold(-1, k, 1)  # (b, c, seq, k): the k causal taps per position
        tpos = (
            torch.arange(seq, device=x.device)[:, None]
            - (k - 1)
            + torch.arange(k, device=x.device)[None, :]
        )  # (seq, k): absolute source position of each tap
        valid = tpos >= 0
        tap_doc = doc_ids[:, tpos.clamp(0, seq - 1)]  # (b, seq, k)
        same = (tap_doc == doc_ids[:, :, None]) & valid[None]  # (b, seq, k)
        win = win * same[:, None]  # zero cross-document / left-pad taps
        w = self.conv1d.weight.squeeze(1)  # (c, k)
        y = (win * w[None, :, None, :]).sum(-1)  # (b, c, seq)
        return F.silu(y).transpose(1, 2)

    def _split_heads(self, qkv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        b, seq = q.shape[:2]
        q = q.reshape(b, seq, self.n_kv_heads, self.d_head)
        k = k.reshape(b, seq, self.n_kv_heads, self.d_head)
        v = v.reshape(b, seq, self.n_heads, self.d_head)
        if self.n_rep > 1:
            q = q.repeat_interleave(self.n_rep, dim=2)
            k = k.repeat_interleave(self.n_rep, dim=2)
        return q, k, v

    def _output(self, core: Tensor, z: Tensor) -> Tensor:
        # core (b, seq, n_heads, d_head); z the output gate in the same layout.
        b, seq = core.shape[:2]
        z = z.reshape(b, seq, self.n_heads, self.d_head)
        out = self.norm(core, z).reshape(b, seq, self.value_dim)
        return self.proj_o(out)

    def forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor | None = None,
        doc_ids: Tensor | None = None,
    ) -> Tensor:
        # Training path. cu_seqlens (1D, into the flattened batch*seq) marks the
        # document/row boundaries at which the recurrent state must reset; None
        # runs each batch row as one sequence. doc_ids (b, seq) resets the short
        # conv at those same boundaries (see _conv). The short conv is applied
        # per row before any flattening.
        b, seq, _ = x.shape
        z = self.proj_z(x)
        qkv = torch.cat([self.proj_q(x), self.proj_k(x), self.proj_v(x)], dim=-1)
        qkv = self._conv(qkv, doc_ids)
        q, k, v = self._split_heads(qkv)
        g, bg, w = self._gates(x)

        if cu_seqlens is not None:
            # Flatten rows to one sequence; segment boundaries (which include the
            # row boundaries) reset the state.
            def flat(u: Tensor) -> Tensor:
                return u.reshape(1, b * seq, *u.shape[2:])

            if _HAS_FLA and x.is_cuda:
                core, _ = _fla_chunk_gdn2(
                    flat(q),
                    flat(k),
                    flat(v),
                    flat(g),
                    flat(bg),
                    flat(w),
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core = _segmented_chunk(
                    flat(q),
                    flat(k),
                    flat(v),
                    flat(g),
                    flat(bg),
                    flat(w),
                    cu_seqlens,
                    self.chunk_size,
                )
            core = core.reshape(b, seq, self.n_heads, self.d_head)
        elif _HAS_FLA and x.is_cuda:
            core, _ = _fla_chunk_gdn2(q, k, v, g, bg, w, use_qk_l2norm_in_kernel=True)
        else:
            core, _ = chunk_gdn2(q, k, v, g, bg, w, chunk_size=self.chunk_size)
        return self._output(core, z)

    def decode(
        self, x: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # Inference path. `state` is (recurrent_state (b, n_heads, d_head,
        # d_head), conv_state (b, conv_dim, conv_size-1)); None starts fresh.
        b, seq, _ = x.shape
        rec_state = None if state is None else state[0]
        conv_state = (
            x.new_zeros(b, self.conv_dim, self.conv_size - 1)
            if state is None
            else state[1]
        )
        z = self.proj_z(x)
        qkv = torch.cat([self.proj_q(x), self.proj_k(x), self.proj_v(x)], dim=-1)
        # Prepend the cached conv context, run an unpadded conv, and remember the
        # last conv_size-1 raw inputs for the next step.
        raw = qkv.transpose(1, 2)  # (b, conv_dim, seq)
        padded = torch.cat([conv_state, raw], dim=-1)
        y = F.silu(F.conv1d(padded, self.conv1d.weight, groups=self.conv_dim))
        new_conv_state = padded[..., -(self.conv_size - 1) :]
        qkv = y.transpose(1, 2)

        q, k, v = self._split_heads(qkv)
        g, bg, w = self._gates(x)
        if _HAS_FLA and x.is_cuda and seq == 1:
            core, rec_state = _fla_recurrent_gdn2(
                q,
                k,
                v,
                g,
                bg,
                w,
                initial_state=rec_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core, rec_state = recurrent_gdn2(q, k, v, g, bg, w, initial_state=rec_state)
        return self._output(core, z), (rec_state, new_conv_state)
