"""Native Sparse Attention (NSA): the sparse softmax mixer that replaces full
(global) attention in picochat's hybrid stack, for long context at a fraction of
the KV cost.

NSA (arXiv:2502.11089, DeepSeek) computes attention through three parallel
branches combined by a per-head learned sigmoid gate (paper Eq. 5):

    o_t = g_t^cmp Attn(q_t, K_cmp) + g_t^slc Attn(q_t, K_slc) + g_t^win Attn(q_t, K_win)

- compressed: keys/values pooled into block reps by a small MLP (with an
  intra-block position encoding), so the query attends over ~T/stride coarse
  tokens;
- selected: the *raw* tokens of the top-n most important blocks, where block
  importance reuses the compressed-branch attention scores (so the selection
  scorer is trained for free through the compressed branch -- the "natively
  trainable" trick; the discrete top-n itself gets no gradient), shared across
  the query heads of a GQA group;
- window: plain local attention over the last `window` tokens.

Each branch has its own K/V projections (independent, to stop the easy
window-branch gradient from suppressing the other two -- paper §3.3.3); the
query is shared. Partial RoPE is applied to q/k (picochat keeps a positional
signal on these softmax layers; the Gated DeltaNet layers are NoPE).

This module is an exact, differentiable, *dense-masked* pure-PyTorch reference
that runs on CPU: it realizes NSA's selection semantics (and their gradient
behavior) but not its memory/throughput win, which comes from the fla Triton
kernels on CUDA (a follow-up swap behind the same interface). forward() is the
packed training path; decode() caches each branch's raw K/V and re-derives the
compression/selection/window per step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

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
            f = (base ** (torch.linspace(0.0, 1.0, rot // 2).repeat_interleave(2)))[None]
            theta = t / f
            self.register_buffer("sin", theta.sin(), persistent=False)
            self.register_buffer("cos", theta.cos(), persistent=False)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        # x (b, h, l, d_head); pos (l,) absolute positions. Rotates x[..., :rot].
        if self.rot_dim == 0:
            return x
        r = self.rot_dim
        with torch.amp.autocast(device_type="cuda", enabled=False):
            sin = self.sin[pos][None, None]  # (1,1,l,rot)
            cos = self.cos[pos][None, None]
            xr, xp = x[..., :r], x[..., r:]
            xr = xr * cos + _rotate_half(xr) * sin
        return torch.cat([xr, xp], dim=-1)


def _masked_softmax_attend(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: float
) -> Tensor:
    """Scaled-dot-product attention with an explicit boolean `mask` (True =
    attend), returning zeros for any query row that may attend to nothing (an
    early token with no available compressed/selected blocks -- the window
    branch always keeps such tokens covered). q (b,h,lq,d), k/v (b,h,lk,d),
    mask (b,h,lq,lk)."""
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
        n_kv_heads: int | None = None,
        cmp_block: int = 32,
        cmp_stride: int = 16,
        sel_block: int = 64,
        n_selected: int = 16,
        window: int = 512,
        rope_factor: float = 0.25,
        rope_base: float = 1_000_000.0,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert n_heads % self.n_kv_heads == 0
        self.n_rep = n_heads // self.n_kv_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0
        self.scale = self.d_head**-0.5

        self.cmp_block = cmp_block
        self.cmp_stride = cmp_stride
        self.sel_block = sel_block
        self.n_selected = n_selected
        self.window = window

        kv_dim = self.n_kv_heads * self.d_head
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        # Independent K/V projections per branch (paper §3.3.3).
        self.proj_k_cmp = nn.Linear(d_model, kv_dim, bias=False)
        self.proj_v_cmp = nn.Linear(d_model, kv_dim, bias=False)
        self.proj_k_slc = nn.Linear(d_model, kv_dim, bias=False)
        self.proj_v_slc = nn.Linear(d_model, kv_dim, bias=False)
        self.proj_k_win = nn.Linear(d_model, kv_dim, bias=False)
        self.proj_v_win = nn.Linear(d_model, kv_dim, bias=False)
        self.proj_o = nn.Linear(d_model, d_model, bias=False)
        # Per-head gate over the three branches (Eq. 5): sigmoid MLP on x.
        self.gate = nn.Linear(d_model, n_heads * 3, bias=False)

        # Compression MLP phi (separate for k and v), with an intra-block
        # position encoding added before pooling: (block_len * d_head) -> d_head.
        self.cmp_pos = nn.Parameter(torch.zeros(cmp_block, self.d_head))
        self.phi_k = nn.Linear(cmp_block * self.d_head, self.d_head, bias=False)
        self.phi_v = nn.Linear(cmp_block * self.d_head, self.d_head, bias=False)

        self.rope = PartialRoPE(self.d_head, rope_factor, rope_base, max_seq_len)

    # -- projections ------------------------------------------------------
    def _project(self, x: Tensor, proj: nn.Linear, heads: int) -> Tensor:
        return rearrange(proj(x), "b l (h d) -> b h l d", h=heads)

    def _q(self, x: Tensor, pos: Tensor) -> Tensor:
        return self.rope(self._project(x, self.proj_q, self.n_heads), pos)

    def _kv(self, x: Tensor, kp: nn.Linear, vp: nn.Linear) -> tuple[Tensor, Tensor]:
        return (
            self._project(x, kp, self.n_kv_heads),
            self._project(x, vp, self.n_kv_heads),
        )

    def _rep(self, t: Tensor) -> Tensor:
        # GQA: expand n_kv_heads -> n_heads along the head dim.
        return t.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else t

    # -- compression ------------------------------------------------------
    def _compress(
        self, k: Tensor, v: Tensor, k_pos: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        # Strided block compression of RoPE'd keys / raw values. Returns
        # (cmp_k, cmp_v, block_last_pos, block_first_pos) or None if the
        # sequence is shorter than one block. k/v: (b, n_kv, T, d_head).
        b, h, T, d = k.shape
        bl, s = self.cmp_block, self.cmp_stride
        if T < bl:
            return None
        starts = torch.arange(0, T - bl + 1, s, device=k.device)
        # gather blocks: (b, h, n_blocks, bl, d)
        idx = starts[:, None] + torch.arange(bl, device=k.device)[None, :]  # (nb, bl)
        kb = k[:, :, idx]  # (b, h, nb, l, d)
        vb = v[:, :, idx]
        kb = kb + self.cmp_pos  # intra-block position encoding
        vb = vb + self.cmp_pos
        cmp_k = self.phi_k(kb.flatten(-2))  # (b, h, nb, d)
        cmp_v = self.phi_v(vb.flatten(-2))
        block_last = starts + (bl - 1)
        return cmp_k, cmp_v, block_last, starts

    # -- core (shared by forward and decode) ------------------------------
    def _attend(
        self,
        x_q: Tensor,
        q: Tensor,
        k_cmp: Tensor,
        v_cmp: Tensor,
        k_slc: Tensor,
        v_slc: Tensor,
        k_win: Tensor,
        v_win: Tensor,
        q_pos: Tensor,
        k_pos: Tensor,
        doc_q: Tensor,
        doc_k: Tensor,
    ) -> Tensor:
        # q (b, H, Lq, d); branch keys/values (b, n_kv, Lk, d) already RoPE'd
        # (keys) where applicable. q_pos/k_pos absolute positions; doc_q/doc_k
        # (b, L) document ids for packing (equal everywhere when unpacked).
        b, H, Lq, d = q.shape
        same_doc = doc_q[:, None, :, None] == doc_k[:, None, None, :]  # (b,1,Lq,Lk)
        causal = k_pos[None, None, None, :] <= q_pos[None, None, :, None]

        # --- window branch ---
        in_win = k_pos[None, None, None, :] > q_pos[None, None, :, None] - self.window
        win_mask = causal & in_win & same_doc
        out_win = _masked_softmax_attend(
            q, self._rep(k_win), self._rep(v_win), win_mask, self.scale
        )

        # --- compressed branch (also produces the selection importances) ---
        comp = self._compress(k_cmp, v_cmp, k_pos)
        out_cmp = torch.zeros_like(out_win)
        p_cmp = None
        blk_first_pos = None
        if comp is not None:
            cmp_k, cmp_v, blk_last, blk_first = comp
            # a compressed block is visible to query t iff it is fully causal
            # and lies entirely within t's document
            blk_last_pos = k_pos[blk_last]  # (nb,)
            blk_first_pos = k_pos[blk_first]
            blk_doc = doc_k[:, blk_last]  # (b, nb) doc of block's last token
            blk_doc_first = doc_k[:, blk_first]
            no_straddle = (blk_doc == blk_doc_first)[:, None, None, :]  # (b,1,1,nb)
            blk_causal = blk_last_pos[None, None, None, :] <= q_pos[None, None, :, None]
            blk_same = blk_doc[:, None, None, :] == doc_q[:, None, :, None]
            blk_mask = blk_causal & blk_same & no_straddle  # (b,1,Lq,nb)

            scores = (q @ self._rep(cmp_k).transpose(-1, -2)) * self.scale  # (b,H,Lq,nb)
            scores = scores.masked_fill(~blk_mask, NEG)
            p_cmp = torch.nan_to_num(scores.softmax(-1), nan=0.0)  # (b,H,Lq,nb)
            out_cmp = p_cmp @ self._rep(cmp_v)

        # --- selected branch (always runs; forced sink+local blocks make it
        # attend to at least the local tokens even when the sequence is shorter
        # than a compression block and there are no importance scores yet) ---
        out_slc = self._select_and_attend(
            q, k_slc, v_slc, p_cmp, blk_first_pos,
            q_pos, k_pos, doc_q, doc_k, causal, same_doc,
        )

        # --- gate & combine ---
        g = self.gate(x_q).sigmoid()  # (b, Lq, H*3)
        g = rearrange(g, "b l (h c) -> c b h l 1", c=3)
        out = g[0] * out_cmp + g[1] * out_slc + g[2] * out_win
        return self.proj_o(rearrange(out, "b h l d -> b l (h d)"))

    def _select_and_attend(
        self, q, k_slc, v_slc, p_cmp, blk_first_pos,
        q_pos, k_pos, doc_q, doc_k, causal, same_doc,
    ) -> Tensor:
        # Map compressed-block importance onto selection blocks, pick top-n per
        # GQA group (plus forced sink + local blocks), and attend over the raw
        # tokens of the selected blocks. `p_cmp`/`blk_first_pos` are None when
        # the sequence is shorter than a compression block (forced-only
        # selection). Returns (b, H, Lq, d).
        b, H, Lq, d = q.shape
        Lk = k_slc.shape[2]
        sb = self.sel_block
        # k_pos is always a contiguous 0..Lk-1, so the number of selection blocks
        # follows from the (static, compile-friendly) key length -- no .item()
        # sync / graph break, which would otherwise block torch.compile fusion.
        n_sel = max(1, (Lk + sb - 1) // sb)

        # per-group selection importance over selection blocks; zero when there
        # are no compressed scores yet
        imp = q.new_zeros(b, self.n_kv_heads, Lq, n_sel)
        if p_cmp is not None:
            # assign each compressed block to the selection block containing its
            # first token, scatter-sum p_cmp, then share across the GQA group
            sel_of_cmp = (blk_first_pos // sb).clamp_max(n_sel - 1)  # (nb,)
            imp_h = q.new_zeros(b, H, Lq, n_sel)
            imp_h.index_add_(3, sel_of_cmp, p_cmp)
            imp = rearrange(imp_h, "b (kv r) l n -> b kv r l n", r=self.n_rep).sum(2)

        # selection-block availability for each query (causal + same doc)
        sel_pos = torch.arange(n_sel, device=q.device) * sb  # block start pos
        sel_id = torch.arange(Lk, device=q.device) // sb  # token -> sel block
        # deterministic tiebreak: nudge importance by block index so exact
        # ties (e.g. all-zero importance) resolve identically regardless of the
        # tensor width -- torch.topk leaves tie order unspecified otherwise.
        # Bounded (< 1e-6 total, regardless of sequence length) so the nudge
        # can never override a genuine importance difference; an unbounded
        # position-proportional nudge would grow past real score gaps on long
        # sequences and bias selection toward late blocks.
        tie = torch.arange(n_sel, device=q.device, dtype=imp.dtype)
        imp = imp + (1e-6 / n_sel) * tie
        # a sel block is available if its first token is causal & same doc as q
        blk_start_causal = sel_pos[None, None, :] <= q_pos[None, :, None]  # (1,Lq,n_sel)
        avail = blk_start_causal.expand(b, Lq, n_sel).clone()
        imp = imp.masked_fill(~avail[:, None], NEG)

        # forced blocks: sink (0), current block, and previous block (always
        # kept -- attention sink + local recency). Up to 3 distinct per query.
        cur = (q_pos // sb).long()  # (Lq,)
        forced = torch.zeros(b, imp.shape[1], Lq, n_sel, dtype=torch.bool, device=q.device)
        forced[..., 0] = True
        ar = torch.arange(Lq, device=q.device)
        forced[:, :, ar, cur] = True
        forced[:, :, ar, (cur - 1).clamp_min(0)] = True
        forced &= avail[:, None]

        # Reserve the (<=3) forced slots and fill the rest competitively from the
        # non-forced available blocks by importance. Keeping the forced blocks
        # out of the topk avoids +inf ties, whose ordering torch.topk leaves
        # unspecified -- that made selection depend on padded (future) blocks and
        # broke causal invariance. Total selected blocks stay <= n_selected.
        n = min(self.n_selected, n_sel)
        n_compete = max(0, n - 3)
        sel_mask_blk = forced.clone()
        if n_compete > 0:
            imp_compete = imp.masked_fill(forced | ~avail[:, None], NEG)
            topk = imp_compete.topk(min(n_compete, n_sel), dim=-1).indices
            picked = torch.zeros_like(imp, dtype=torch.bool)
            picked.scatter_(-1, topk, True)
            picked &= imp_compete > NEG  # drop padded (all-masked) picks
            sel_mask_blk |= picked

        # expand selection-block mask to token level and to all heads
        tok_mask = sel_mask_blk[..., sel_id]  # (b, kv, Lq, Lk)
        tok_mask = tok_mask.repeat_interleave(self.n_rep, dim=1)  # (b, H, Lq, Lk)
        tok_mask = tok_mask & causal & same_doc
        return _masked_softmax_attend(
            q, self._rep(k_slc), self._rep(v_slc), tok_mask, self.scale
        )

    # -- forward / decode -------------------------------------------------
    def forward(self, x: Tensor, doc_ids: Tensor | None = None) -> Tensor:
        # Training path: q positions == k positions == 0..L-1.
        b, L, _ = x.shape
        pos = torch.arange(L, device=x.device)
        doc = (
            torch.zeros(b, L, dtype=torch.long, device=x.device)
            if doc_ids is None
            else doc_ids
        )
        q = self._q(x, pos)
        k_cmp, v_cmp = self._kv(x, self.proj_k_cmp, self.proj_v_cmp)
        k_slc, v_slc = self._kv(x, self.proj_k_slc, self.proj_v_slc)
        k_win, v_win = self._kv(x, self.proj_k_win, self.proj_v_win)
        k_cmp = self.rope(k_cmp, pos)
        k_slc = self.rope(k_slc, pos)
        k_win = self.rope(k_win, pos)
        return self._attend(
            x, q, k_cmp, v_cmp, k_slc, v_slc, k_win, v_win, pos, pos, doc, doc
        )

    def decode(
        self, x: Tensor, cache: dict | None = None, pos: int = 0
    ) -> tuple[Tensor, dict]:
        # Inference path: cache each branch's raw (RoPE'd) K / raw V over the
        # full history; recompute compression / selection / window per step. The
        # cache grows in the sequence dim, so speculative-decode rollback can
        # slice it like an ordinary KV cache.
        b, L, _ = x.shape
        q_pos = torch.arange(pos, pos + L, device=x.device)
        q = self._q(x, q_pos)
        kc, vc = self._kv(x, self.proj_k_cmp, self.proj_v_cmp)
        ks, vs = self._kv(x, self.proj_k_slc, self.proj_v_slc)
        kw, vw = self._kv(x, self.proj_k_win, self.proj_v_win)
        kc, ks, kw = self.rope(kc, q_pos), self.rope(ks, q_pos), self.rope(kw, q_pos)
        if cache is not None:
            kc = torch.cat([cache["kc"], kc], dim=2)
            vc = torch.cat([cache["vc"], vc], dim=2)
            ks = torch.cat([cache["ks"], ks], dim=2)
            vs = torch.cat([cache["vs"], vs], dim=2)
            kw = torch.cat([cache["kw"], kw], dim=2)
            vw = torch.cat([cache["vw"], vw], dim=2)
        new_cache = {"kc": kc, "vc": vc, "ks": ks, "vs": vs, "kw": kw, "vw": vw}
        k_pos = torch.arange(kw.shape[2], device=x.device)
        doc = torch.zeros(b, kw.shape[2], dtype=torch.long, device=x.device)
        out = self._attend(
            x, q, kc, vc, ks, vs, kw, vw, q_pos, k_pos, doc[:, -L:], doc
        )
        return out, new_cache
