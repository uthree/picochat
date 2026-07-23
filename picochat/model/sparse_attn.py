"""Native Sparse Attention (NSA): the sparse softmax mixer that replaces full
(global) attention in picochat's hybrid stack, for long context at a fraction of
the KV cost.

NSA (arXiv:2502.11089, DeepSeek) computes attention through three parallel
branches combined by a per-head learned sigmoid gate (paper Eq. 5):

    o_t = g_t^cmp Attn(q_t, K_cmp) + g_t^slc Attn(q_t, K_slc) + g_t^win Attn(q_t, K_win)

- compressed: keys/values mean-pooled into non-overlapping `block_size` blocks,
  so the query attends over ~T/block_size coarse tokens;
- selected: the *raw* tokens of the top-n most important blocks, where block
  importance reuses the compressed-branch attention scores (so the selection
  scorer is trained for free through the compressed branch -- the "natively
  trainable" trick; the discrete top-n itself gets no gradient), shared across
  the query heads of a GQA group. The sink block (0) and the current/previous
  blocks are always selected;
- window: plain local attention over the last `window` tokens.

The semantics follow fla's Triton NSA kernels exactly (`fla.ops.nsa`), so the
training path can run on them on CUDA: `parallel_nsa` fuses the compressed
branch, the top-n selection and the selected attention (no O(T^2) score
materialization), and `parallel_attn` (also fla) provides the sliding-window
branch. Both accept `cu_seqlens` for sequence packing, treating each document
as its own sequence (block structure resets per document). This is what fla's
kernel design implies for the module, vs. the paper's full generality: one
shared K/V per layer (not per-branch), mean-pooling compression (not a learned
MLP), and compression block == selection block with no overlap.

The fla kernels additionally require the GQA group size (n_heads / n_kv_heads)
to be a multiple of 16 -- the paper's regime (its config has 64 query heads in
GQA-4, group 16), since a whole group is loaded as one Triton tile. Configs
that violate it still work but fall back to the reference path below.

The pure-PyTorch reference (`_reference`) implements the same math dense-masked
for CPU, non-kernel configs and the decode path, which caches raw K/V like an
ordinary KV cache and re-derives pooling/selection/window per step.

Partial RoPE is applied to q/k before everything (picochat keeps a positional
signal on these softmax layers; the Gated DeltaNet layers are NoPE), so both
paths and both branches see identically rotated keys.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

try:  # CUDA Triton kernels (fla-core, a Linux dependency); guarded so
    # macOS/Windows fall back cleanly to the pure-PyTorch reference below.
    from fla.ops.attn.parallel import parallel_attn as _fla_parallel_attn
    from fla.ops.nsa.parallel import parallel_nsa as _fla_parallel_nsa

    _HAS_FLA = True
except Exception:  # pragma: no cover - depends on the environment
    _HAS_FLA = False

NEG = float("-inf")


def _rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    return rearrange(torch.stack([-x2, x1], dim=-1), "... d r -> ... (d r)")


class PartialRoPE(nn.Module):
    """Rotary position embedding applied to only the first `rot_dim` (even)
    channels of each head, leaving the rest un-rotated (NoPE) -- the partial-
    RoPE scheme picochat's NSA layers use. `rot_dim = round(d_head * factor)`."""

    def __init__(self, d_head: int, factor: float, base: float, max_seq_len: int):
        super().__init__()
        rot = int(round(d_head * factor))
        rot -= rot % 2  # RoPE rotates dimension pairs
        self.rot_dim = rot
        self.max_seq_len = max_seq_len
        if rot > 0:
            t = torch.arange(max_seq_len)[:, None].float()
            f = (base ** (torch.linspace(0.0, 1.0, rot // 2).repeat_interleave(2)))[
                None
            ]
            theta = t / f
            self.register_buffer("sin", theta.sin(), persistent=False)
            self.register_buffer("cos", theta.cos(), persistent=False)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        # x (b, l, h, d_head); pos (l,) absolute positions. Rotates x[..., :rot].
        if self.rot_dim == 0:
            return x
        r = self.rot_dim
        with torch.amp.autocast(device_type="cuda", enabled=False):
            sin = self.sin[pos][None, :, None]  # (1, l, 1, rot)
            cos = self.cos[pos][None, :, None]
            xr, xp = x[..., :r].float(), x[..., r:]
            xr = (xr * cos + _rotate_half(xr) * sin).to(x.dtype)
        return torch.cat([xr, xp], dim=-1)


def _masked_softmax_attend(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: float
) -> Tensor:
    """Scaled-dot-product attention with an explicit boolean `mask` (True =
    attend), returning zeros for any query row that may attend to nothing.
    q (b,h,lq,d), k/v (b,h,lk,d), mask broadcastable to (b,h,lq,lk)."""
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask, NEG)
    attn = scores.softmax(-1)
    attn = torch.nan_to_num(attn, nan=0.0)  # all-masked rows -> 0 output
    return attn @ v


class NativeSparseAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int = 1,
        block_size: int = 64,
        n_selected: int = 16,
        window: int = 512,
        rope_factor: float = 0.25,
        rope_base: float = 1_000_000.0,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # NSA's own KV head count (independent of the GDN layers' GQA setting):
        # the paper shares one selection per GQA group, and the fla kernels tile
        # a whole group, requiring n_heads/n_kv_heads to be a multiple of 16.
        self.n_kv_heads = n_kv_heads
        assert n_heads % n_kv_heads == 0
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0
        self.scale = self.d_head**-0.5

        self.block_size = block_size
        self.n_selected = n_selected
        self.window = window

        kv_dim = self.n_kv_heads * self.d_head
        # One shared K/V for all three branches (the fla-kernel factorization;
        # the compressed branch derives its keys/values by mean-pooling these).
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, kv_dim, bias=False)
        self.proj_v = nn.Linear(d_model, kv_dim, bias=False)
        self.proj_o = nn.Linear(d_model, d_model, bias=False)
        # Per-head gate over the three branches (Eq. 5): sigmoid MLP on x.
        self.gate = nn.Linear(d_model, n_heads * 3, bias=False)

        self.rope = PartialRoPE(self.d_head, rope_factor, rope_base, max_seq_len)

    def _use_kernel(self, x: Tensor) -> bool:
        # The fla NSA kernels tile one GQA group per Triton block: the group
        # size must be a multiple of 16, and the top-k tile needs
        # block_size >= 2 * n_selected.
        return (
            _HAS_FLA
            and x.is_cuda
            and self.n_rep % 16 == 0
            and self.block_size >= 2 * self.n_selected
        )

    def _qkv(self, x: Tensor, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, seq, _ = x.shape
        q = self.proj_q(x).reshape(b, seq, self.n_heads, self.d_head)
        k = self.proj_k(x).reshape(b, seq, self.n_kv_heads, self.d_head)
        v = self.proj_v(x).reshape(b, seq, self.n_kv_heads, self.d_head)
        return self.rope(q, pos), self.rope(k, pos), v

    def _gates(self, x: Tensor) -> Tensor:
        # (b, l, n_heads, 3): sigmoid gate per query head per branch,
        # ordered (cmp, slc, win).
        return rearrange(self.gate(x), "b l (h c) -> b l h c", c=3).sigmoid()

    def _rep(self, t: Tensor) -> Tensor:
        # GQA: expand n_kv_heads -> n_heads along the head dim (b, h, l, d).
        return t.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else t

    # -- reference path ----------------------------------------------------
    def _reference(
        self, q: Tensor, k: Tensor, v: Tensor, g: Tensor, q_pos: Tensor
    ) -> Tensor:
        """Dense-masked pure-PyTorch NSA over ONE sequence segment, matching
        fla's kernel semantics exactly (see module docstring). q (b, Lq, HQ, d)
        and g (b, Lq, HQ, 3) cover the query positions `q_pos` (0-based within
        the segment); k/v (b, Lk, H, d) cover key positions 0..Lk-1 -- Lq == Lk
        for training, a suffix of the timeline for decode. Returns
        (b, Lq, HQ, d)."""
        b, lq, hq, dh = q.shape
        lk, h = k.shape[1], k.shape[2]
        bs, s = self.block_size, self.n_selected
        qh = q.permute(0, 2, 1, 3)  # (b, HQ, Lq, d)
        kh = k.permute(0, 2, 1, 3)  # (b, H, Lk, d)
        vh = v.permute(0, 2, 1, 3)
        k_pos = torch.arange(lk, device=q.device)

        # --- window branch: keys in (t - window, t] ---
        rel = q_pos[:, None] - k_pos[None, :]  # (Lq, Lk)
        win_mask = (rel >= 0) & (rel < self.window)
        out_win = _masked_softmax_attend(
            qh, self._rep(kh), self._rep(vh), win_mask[None, None], self.scale
        )

        # --- compressed branch: mean-pooled complete blocks; block j becomes
        # visible at its last token ((j+1)*bs - 1 <= t, i.e. j < (t+1)//bs) ---
        nb = lk // bs  # complete blocks only
        n_all = (lk + bs - 1) // bs  # incl. the trailing partial block
        imp = qh.new_zeros(b, h, lq, n_all)  # selection importance per KV head
        out_cmp = torch.zeros_like(out_win)
        if nb > 0:
            kc = kh[:, :, : nb * bs].reshape(b, h, nb, bs, dh).mean(3)
            vc = vh[:, :, : nb * bs].reshape(b, h, nb, bs, dh).mean(3)
            s_cmp = (qh @ self._rep(kc).transpose(-1, -2)) * self.scale  # (b,HQ,Lq,nb)
            ncq = (q_pos + 1) // bs  # visible complete blocks per query
            vis = torch.arange(nb, device=q.device)[None, :] < ncq[:, None]  # (Lq, nb)
            s_cmp = s_cmp.masked_fill(~vis[None, None], NEG)
            # p == exp(s - lse): the softmax over visible blocks; a query with
            # no visible block gets all-zero p (fla defines lse = 0 there).
            p_cmp = torch.nan_to_num(s_cmp.softmax(-1), nan=0.0)
            out_cmp = p_cmp @ self._rep(vc)
            # importance = p summed over the GQA group (selection is shared
            # across the group's query heads)
            imp[..., :nb] = rearrange(p_cmp, "b (h g) l n -> b h g l n", h=h).sum(2)

        # --- selection: top-n blocks per KV head among blocks 0..cur, with the
        # sink (0), previous and current blocks forced to maximal importance
        # (fla scores them 1.0 per query head, i.e. n_rep after the group sum,
        # >= any real score sum) ---
        cur = (q_pos // bs).long()  # (Lq,)
        blk = torch.arange(n_all, device=q.device)
        cand = blk[None, :] <= cur[:, None]  # (Lq, n_all) selectable blocks
        ar = torch.arange(lq, device=q.device)
        forced = torch.zeros(lq, n_all, dtype=torch.bool, device=q.device)
        forced[:, 0] = True
        forced[ar, cur] = True
        forced[ar, (cur - 1).clamp_min(0)] = True
        imp = imp.masked_fill(forced[None, None], float(self.n_rep))
        imp = imp.masked_fill(~cand[None, None], NEG)
        topk = imp.topk(min(s, n_all), dim=-1)
        picked = torch.zeros_like(imp, dtype=torch.bool)
        picked.scatter_(-1, topk.indices, True)
        picked &= imp > NEG  # drop picks that were only masked padding

        # expand the block mask to token level, all heads, causal
        sel_id = (k_pos // bs).clamp_max(n_all - 1)  # token -> block
        tok_mask = picked[..., sel_id]  # (b, H, Lq, Lk)
        tok_mask = tok_mask.repeat_interleave(self.n_rep, dim=1)
        tok_mask = tok_mask & (k_pos[None, None, None, :] <= q_pos[None, None, :, None])
        out_slc = _masked_softmax_attend(
            qh, self._rep(kh), self._rep(vh), tok_mask, self.scale
        )

        gh = g.permute(0, 2, 1, 3)[..., None]  # (b, HQ, Lq, 3, 1) -> slices below
        out = (
            gh[..., 0, :] * out_cmp + gh[..., 1, :] * out_slc + gh[..., 2, :] * out_win
        )
        return out.permute(0, 2, 1, 3)  # (b, Lq, HQ, d)

    def _reference_segments(
        self, q: Tensor, k: Tensor, v: Tensor, g: Tensor, cu_seqlens: Tensor
    ) -> Tensor:
        # Packed training on the reference path: run each cu_seqlens segment
        # as its own sequence (positions and block structure restart per
        # document, exactly like fla's varlen mode). Inputs are (1, T, ...).
        outs = []
        for s0, e0 in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            t = e0 - s0
            pos = torch.arange(t, device=q.device)
            outs.append(
                self._reference(q[:, s0:e0], k[:, s0:e0], v[:, s0:e0], g[:, s0:e0], pos)
            )
        return torch.cat(outs, dim=1)

    # -- forward / decode -------------------------------------------------
    def forward(
        self,
        x: Tensor,
        doc_ids: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor:
        # Training path. Sequence packing comes in as `cu_seqlens` (1D bounds
        # into the flattened batch*seq, always including row boundaries -- see
        # doc_ids_to_cu_seqlens); each segment is treated as an independent
        # sequence. `doc_ids` is accepted for interface symmetry but unused --
        # cu_seqlens carries the same information here.
        b, seq, _ = x.shape
        pos = torch.arange(seq, device=x.device)
        q, k, v = self._qkv(x, pos)  # RoPE'd per row (relative offsets survive
        # flattening: RoPE is relative under dot products)
        g = self._gates(x)

        if self._use_kernel(x):
            if cu_seqlens is not None:
                q, k, v, g = (u.reshape(1, b * seq, *u.shape[2:]) for u in (q, k, v, g))
            o = _fla_parallel_nsa(
                q,
                k,
                v,
                g_cmp=g[..., 0].contiguous(),
                g_slc=g[..., 1].contiguous(),
                block_counts=self.n_selected,
                block_size=self.block_size,
                scale=self.scale,
                cu_seqlens=cu_seqlens,
            )
            o_win = _fla_parallel_attn(
                q,
                k,
                v,
                scale=self.scale,
                window_size=self.window,
                cu_seqlens=cu_seqlens,
            )
            o = o + g[..., 2, None] * o_win
            o = o.reshape(b, seq, self.d_model)
        elif cu_seqlens is not None:
            flat = (u.reshape(1, b * seq, *u.shape[2:]) for u in (q, k, v, g))
            o = self._reference_segments(*flat, cu_seqlens).reshape(
                b, seq, self.d_model
            )
        else:
            o = self._reference(q, k, v, g, pos).reshape(b, seq, self.d_model)
        return self.proj_o(o)

    def decode(
        self, x: Tensor, cache: dict | None = None, pos: int = 0
    ) -> tuple[Tensor, dict]:
        # Inference path: cache the raw (RoPE'd) K / raw V over the full
        # history and re-derive pooling / selection / window per step. The
        # cache grows in the sequence dim, so speculative-decode rollback can
        # snapshot/restore it like an ordinary KV cache.
        b, seq, _ = x.shape
        q_pos = torch.arange(pos, pos + seq, device=x.device)
        q, k, v = self._qkv(x, q_pos)
        if cache is not None:
            k = torch.cat([cache["k"], k], dim=1)
            v = torch.cat([cache["v"], v], dim=1)
        new_cache = {"k": k, "v": v}
        g = self._gates(x)
        if cache is None and self._use_kernel(x):
            # Fresh-prefill fast path: with no history this is exactly the
            # training forward, so the O(T^2)-free kernels handle long prompts.
            o = _fla_parallel_nsa(
                q,
                k,
                v,
                g_cmp=g[..., 0].contiguous(),
                g_slc=g[..., 1].contiguous(),
                block_counts=self.n_selected,
                block_size=self.block_size,
                scale=self.scale,
            )
            o = o + g[..., 2, None] * _fla_parallel_attn(
                q, k, v, scale=self.scale, window_size=self.window
            )
            o = o.reshape(b, seq, self.d_model)
        else:
            o = self._reference(q, k, v, g, q_pos).reshape(b, seq, self.d_model)
        return self.proj_o(o), new_cache
