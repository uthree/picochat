"""Gated DeltaNet: the linear-attention (recurrent) mixer that replaces sliding-
window softmax attention in picochat's hybrid stack.

Gated DeltaNet (arXiv:2412.06464, "Gated Delta Networks: Improving Mamba2 with
Delta Rule") maintains a fixed-size matrix state S_t per head instead of a
growing KV cache, updated per token by a *gated delta rule* -- Mamba2's scalar
forget gate a_t combined with DeltaNet's delta write of strength b_t:

    S_t = S_{t-1} (a_t (I - b_t k_t k_t^T)) + b_t v_t k_t^T,   o_t = S_t q_t

The recurrence is inherently order-dependent, so these layers carry position
information implicitly and use *no* RoPE (the softmax layers -- picochat's NSA
layers -- keep their own positional scheme). Two equivalent evaluation paths:

- training: a chunkwise-parallel form (`chunk_gated_delta_rule`) that is
  matmul-heavy and torch.compile-friendly, ported from HF's Qwen3-Next
  reference (which itself follows the paper's WY-representation derivation);
- decode: the plain O(1)-per-token recurrence (`recurrent_gated_delta_rule`).

Both are exact pure-PyTorch and run on CPU (the correctness oracle / fallback).
On CUDA the optional `fla` (flash-linear-attention) Triton kernels are used
when installed -- see `_fla_chunk`. Sequence packing resets the recurrent state
at document boundaries via `cu_seqlens` (segment-wise here; native in fla).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:  # optional CUDA acceleration; everything works without it
    from fla.ops.gated_delta_rule import (  # type: ignore
        chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule as _fla_recurrent_gated_delta_rule,
    )

    _HAS_FLA = True
except Exception:  # pragma: no cover - depends on the environment
    _HAS_FLA = False


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """L2-normalize along `dim` (the feature-map normalization the paper's
    ablation found best for q/k; matches fla's kernel-side l2norm)."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[Tensor, Tensor]:
    """Sequential gated delta rule (the O(1)-per-token decode path).

    Shapes: query/key (B, T, H, Dk), value (B, T, H, Dv), g/beta (B, T, H)
    with g the *log* forget gate (<= 0) and beta in (0, 1). Returns the outputs
    (B, T, H, Dv) and the final state (B, H, Dk, Dv). Runs in float32 for
    stability. Ported from HF Qwen3-Next's torch_recurrent_gated_delta_rule.
    """
    dtype = query.dtype
    if use_qk_l2norm:
        query, key = l2norm(query), l2norm(key)
    query, key, value, beta, g = (
        t.transpose(1, 2).contiguous().float() for t in (query, key, value, beta, g)
    )
    b, h, t, dk = key.shape
    dv = value.shape[-1]
    query = query * dk**-0.5

    out = torch.zeros(b, h, t, dv, dtype=torch.float32, device=value.device)
    state = (
        torch.zeros(b, h, dk, dv, dtype=torch.float32, device=value.device)
        if initial_state is None
        else initial_state.float()
    )
    for i in range(t):
        k_t = key[:, :, i]  # (b, h, dk)
        v_t = value[:, :, i]  # (b, h, dv)
        state = state * g[:, :, i].exp()[..., None, None]  # decay a_t
        kv_mem = (state * k_t[..., None]).sum(-2)  # k_t^T S  (b, h, dv)
        delta = (v_t - kv_mem) * beta[:, :, i][..., None]  # b_t (v - k^T S)
        state = state + k_t[..., None] * delta[..., None, :]  # + k (.)^T
        out[:, :, i] = (state * query[:, :, i][..., None]).sum(-2)  # q^T S
    return out.transpose(1, 2).contiguous().to(dtype), state


def chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
) -> tuple[Tensor, Tensor]:
    """Chunkwise-parallel gated delta rule (the training path): exactly equal to
    the sequential recurrence but expressed as batched matmuls over length-C
    chunks via the gated WY representation (paper Eqs. 9-12). Same shapes /
    return contract as recurrent_gated_delta_rule. Ported from HF Qwen3-Next's
    torch_chunk_gated_delta_rule.
    """
    dtype = query.dtype
    if use_qk_l2norm:
        query, key = l2norm(query), l2norm(key)
    query, key, value, beta, g = (
        t.transpose(1, 2).contiguous().float() for t in (query, key, value, beta, g)
    )
    b, h, t, dk = key.shape
    dv = value.shape[-1]
    # The chunk size only sets the parallelization granularity, not the result
    # (exactly equal to the sequential recurrence for any value). Cap it at the
    # sequence length so short sequences don't pay the full-chunk Python loops.
    chunk_size = max(1, min(chunk_size, t))
    pad = (chunk_size - t % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    tt = t + pad
    query = query * dk**-0.5

    v_beta = value * beta[..., None]
    k_beta = key * beta[..., None]
    c = chunk_size
    query, key, value, k_beta, v_beta = (
        x.reshape(b, h, -1, c, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(b, h, -1, c)
    tri = torch.triu(torch.ones(c, c, dtype=torch.bool, device=query.device), 0)

    g = g.cumsum(-1)
    decay = (g[..., :, None] - g[..., None, :]).tril().exp().tril()
    # (I - tril(diag(beta) K K^T))^{-1}, built by forward substitution.
    attn = -((k_beta @ key.transpose(-1, -2)) * decay).masked_fill(tri, 0)
    for i in range(1, c):
        row = attn[..., i, :i].clone()
        attn[..., i, :i] = row + (row[..., None] * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(c, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta  # pseudo-values U
    k_cumdecay = attn @ (k_beta * g.exp()[..., None])

    state = (
        torch.zeros(b, h, dk, dv, dtype=torch.float32, device=value.device)
        if initial_state is None
        else initial_state.float()
    )
    out = torch.zeros_like(value)
    causal = torch.triu(torch.ones(c, c, dtype=torch.bool, device=query.device), 1)
    for i in range(tt // c):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        a = (q_i @ k_i.transpose(-1, -2) * decay[:, :, i]).masked_fill(causal, 0)
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        inter = (q_i * g[:, :, i, :, None].exp()) @ state
        out[:, :, i] = inter + a @ v_new
        state = state * g[:, :, i, -1, None, None].exp() + (
            k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new

    out = out.reshape(b, h, -1, dv)[:, :, :t]
    return out.transpose(1, 2).contiguous().to(dtype), state


def _segmented_chunk(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    cu_seqlens: Tensor,
    chunk_size: int,
) -> Tensor:
    """Run the chunk rule independently on each `cu_seqlens` segment so the
    recurrent state resets at every document boundary (sequence packing). Inputs
    are flattened to a single row (B=1, T=total); returns (1, T, H, Dv). Used by
    the pure-PyTorch path; fla accepts cu_seqlens natively."""
    outs = []
    for s, e in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        o, _ = chunk_gated_delta_rule(
            query[:, s:e],
            key[:, s:e],
            value[:, s:e],
            g[:, s:e],
            beta[:, s:e],
            chunk_size=chunk_size,
        )
        outs.append(o)
    return torch.cat(outs, dim=1)


class GatedRMSNorm(nn.Module):
    """RMSNorm with a SiLU output gate: weight * rms_norm(x) * silu(gate). The
    per-head output normalization + gate at the end of a Gated DeltaNet mixer
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


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet mixer (drop-in replacement for a sliding-window
    SelfAttention layer). Mirrors picochat's GQA head convention: `n_heads`
    value/output heads and `n_kv_heads` key heads (n_heads a multiple of
    n_kv_heads; q/k are repeat-interleaved up to n_heads for the recurrence),
    with a square head dim `d_model // n_heads` (so the value dim is d_model and
    out_proj is square). No RoPE -- positions come from the recurrence itself.

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
        assert n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
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
        # a/b: one scalar per value-head per token. a feeds the Mamba2 forget
        # gate, b the delta write strength.
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)
        # Mamba2 gate parameterization: g = -exp(A_log) * softplus(a + dt_bias).
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))
        self.A_log = nn.Parameter(torch.zeros(n_heads))
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
            self.A_log.copy_(torch.log(torch.empty(self.n_heads).uniform_(1, 16)))
            self.dt_bias.zero_()
            self.norm.weight.fill_(1.0)

    def _gates(self, x: Tensor) -> tuple[Tensor, Tensor]:
        beta = self.b_proj(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)
        return g, beta

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
        tpos = torch.arange(seq, device=x.device)[:, None] - (k - 1) + torch.arange(
            k, device=x.device
        )[None, :]  # (seq, k): absolute source position of each tap
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
        g, beta = self._gates(x)

        if cu_seqlens is not None:
            # Flatten rows to one sequence; segment boundaries (which include the
            # row boundaries) reset the state.
            def flat(u: Tensor) -> Tensor:
                return u.reshape(1, b * seq, *u.shape[2:])

            if _HAS_FLA and x.is_cuda:
                core, _ = _fla_chunk_gated_delta_rule(
                    flat(q), flat(k), flat(v), g=flat(g), beta=flat(beta),
                    cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
                )
            else:
                core = _segmented_chunk(
                    flat(q), flat(k), flat(v), flat(g), flat(beta),
                    cu_seqlens, self.chunk_size,
                )
            core = core.reshape(b, seq, self.n_heads, self.d_head)
        elif _HAS_FLA and x.is_cuda:
            core, _ = _fla_chunk_gated_delta_rule(
                q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True
            )
        else:
            core, _ = chunk_gated_delta_rule(
                q, k, v, g, beta, chunk_size=self.chunk_size
            )
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
        g, beta = self._gates(x)
        if _HAS_FLA and x.is_cuda and seq == 1:
            core, rec_state = _fla_recurrent_gated_delta_rule(
                q, k, v, g=g, beta=beta, initial_state=rec_state,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
        else:
            core, rec_state = recurrent_gated_delta_rule(
                q, k, v, g, beta, initial_state=rec_state
            )
        return self._output(core, z), (rec_state, new_conv_state)
