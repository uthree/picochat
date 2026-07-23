"""Model growth / upcycling: warm-start a larger model from a smaller trained one.

Three transforms on a bare TransformerLM state_dict, composed by
`grow_state_dict()` to turn a small model's weights into the initialization of a
larger one -- much cheaper than pretraining each rung from scratch, and the
small model's learned features carry over instead of being thrown away:

  1. grow_width    -- HyperCloning (Samragh et al., 2024): widen d_model by an
                      integer factor r, keeping d_head constant and replicating
                      each attention head r-fold. *Exactly* function-preserving
                      (the widened model computes the identical function at init).
  2. grow_depth    -- append transformer blocks (Llama Pro / gradual-stacking
                      style), each new block seeded from a trained one but with
                      its residual-writing projections zeroed, so it starts as a
                      near-identity. *Approximately* function-preserving -- see
                      the DepthAttention note on grow_depth.
  3. upcycle_to_moe -- sparse upcycling (Komatsuzaki et al., 2022): add a routed
                      MoE branch to every layer, its output projection zeroed so
                      the model starts identical to the dense one. Function-
                      preserving when the dense/shared FFN width is preserved
                      (see the d_ffn note on upcycle_to_moe).

Everything operates on a plain {name: Tensor} state_dict -- the *bare*
TransformerLM keys, i.e. a GPT Lightning checkpoint with the "model." prefix
stripped -- plus normalized config dicts, and allocates only the target tensors:
the (possibly enormous) target model is never built. scripts/base_train.py wires
this in through a config's `grow_from`.

Why constant d_head for width growth: per-head softmax attention is invariant
under widening only if the head dimension is unchanged -- the d_head**-0.5 score
scale and the RoPE frequencies both depend on it -- so widening multiplies the
head *count*, not the head *size*. Every picochat rung on a width-growth chain
must therefore share d_head (see configs/presets.yml). grow_width asserts it.
"""

from __future__ import annotations

import torch
from torch import Tensor

# The per-layer keys whose linear writes the mixer/MLP output back into the
# residual stream. Zeroing these makes a layer contribute nothing at init (the
# basis for grow_depth's near-identity blocks and upcycle_to_moe's zero-init
# expert branch).
_RESIDUAL_WRITE = {
    "attn.proj_o.weight",  # GDN / NSA mixer output
    "ffn.proj_down.weight",  # dense SwiGLU output
    "moe.weight_expand",  # LatentMoE expansion back to d_model
    "moe.bank.weight_down",  # non-latent MoE expert output
}


def _norm_cfg(cfg: dict) -> dict:
    """Fill a preset/model_config's derived defaults so the growth math has
    concrete numbers (mirrors TransformerLM's own argument defaults)."""
    d_model, n_heads = cfg["d_model"], cfg["n_heads"]
    d_ffn = cfg.get("d_ffn") or 3 * d_model
    n_experts = cfg.get("n_experts")
    out = dict(
        vocab_size=cfg["vocab_size"],
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=cfg.get("n_kv_heads") or n_heads,
        nsa_kv_heads=cfg.get("nsa_kv_heads", 1),
        n_layers=cfg["n_layers"],
        layers_per_block=cfg.get("layers_per_block", 4),
        conv_size=cfg.get("conv_size", 4),
        d_ffn=d_ffn,
        n_experts=n_experts,
        n_active=cfg.get("n_active", 2),
        d_expert=(cfg.get("d_expert") or d_ffn) if n_experts is not None else None,
        d_latent=cfg.get("d_latent"),
        share_experts=bool(cfg.get("share_experts", False)),
        n_mtp=cfg.get("n_mtp", 0),
        mtp_rank=cfg.get("mtp_rank"),
        init_std=cfg.get("init_std", 0.02),
    )
    assert d_model % n_heads == 0, "heads must tile d_model"
    out["d_head"] = d_model // n_heads
    return out


# ---------------------------------------------------------------------------
# width (HyperCloning)
# ---------------------------------------------------------------------------
# The widening invariant: a widened activation of dimension r*d represents the
# original d-dim activation `a` as r stacked copies, a' = [a; a; ...; a]. Every
# module must map an r-copied input to an r-copied output for the invariant (and
# thus the function) to be preserved. RMSNorm and every elementwise nonlinearity
# preserve it for free (mean-of-squares and per-element maps are copy-invariant);
# the linear layers below are built to. Because d_head is held fixed, each head's
# attention is bit-for-bit unchanged and the head count simply multiplies -- the
# new heads are copies of the originals.


def _both(W: Tensor, r: int, noise: float, gen: torch.Generator) -> Tensor:
    """Widen a linear whose input AND output are d-widened: W (out, in) ->
    (r*out, r*in), laid out as r x r blocks each equal to W/r. Output copy i is
    sum_j (W/r) x = W x, so the output is r copies of Wx (function-preserving).
    `noise` adds a symmetry-breaking perturbation with each block-row summing to
    zero, so the r copies stop being identical (and can diverge in training)
    while the sum -- and hence the preserved function -- is untouched."""
    out, inn = W.shape
    base = (W / r).repeat(r, r)  # block(i, j) = W / r
    if noise > 0:
        eps = torch.randn(r, r, out, inn, generator=gen, dtype=W.dtype)
        eps = eps * (noise * W.std().clamp_min(1e-8))
        eps = eps - eps.mean(dim=1, keepdim=True)  # sum over input-copies j == 0
        base = base + eps.permute(0, 2, 1, 3).reshape(r * out, r * inn)
    return base


def _in(W: Tensor, r: int, noise: float, gen: torch.Generator) -> Tensor:
    """Widen a linear whose input is d-widened but output is not (lm head,
    MoE router): W (out, in) -> (out, r*in), r horizontal blocks each W/r, so
    W'[a'] = sum_j (W/r) x = W x on an r-copied input a' = [x; ...; x]."""
    out, inn = W.shape
    base = (W / r).repeat(1, r)
    if noise > 0:
        eps = torch.randn(out, r, inn, generator=gen, dtype=W.dtype)
        eps = eps * (noise * W.std().clamp_min(1e-8))
        eps = eps - eps.mean(dim=1, keepdim=True)  # sum over input-copies == 0
        base = base + eps.reshape(out, r * inn)
    return base


def _widen_conv(W: Tensor, r: int, key_dim: int, value_dim: int) -> Tensor:
    """Widen the GDN depthwise short conv (conv_dim, 1, k). conv_dim is
    concat(q, k, v) of widths (key_dim, key_dim, value_dim); after widening each
    projection emits r stacked copies, so each of the three segments is tiled r
    times (channel-copy-major) to line up with that layout."""
    q, k, v = W.split([key_dim, key_dim, value_dim], dim=0)
    return torch.cat([q.repeat(r, 1, 1), k.repeat(r, 1, 1), v.repeat(r, 1, 1)], dim=0)


def _widen_key(
    k: str, W: Tensor, r: int, src: dict, noise: float, gen: torch.Generator
) -> Tensor:
    key_dim = src["n_kv_heads"] * src["d_head"]  # GDN proj_q/k output width
    value_dim = src["d_model"]  # GDN proj_v/z output width
    if k == "embed.weight":
        # The embedding *is* the source of the r-copied activation: tile the
        # feature dim, no 1/r (downstream linears carry the 1/r that keeps the
        # function fixed).
        return W.repeat(1, r)
    if k == "lmhead.weight":
        # Reads the (r-copied) final hidden state; input-widened only.
        return _in(W, r, noise, gen)
    if k.endswith(".query"):
        # DepthAttention query: the logit is a dot product summed over the r*d
        # widened dims, so it picks up a factor r; tile then divide by r to keep
        # the softmax temperature (and thus the residual mix) identical.
        return W.repeat(r) / r
    if k.endswith(".conv1d.weight"):
        return _widen_conv(W, r, key_dim, value_dim)
    if k.endswith(".f_proj.0.weight"):
        # GDN-2 decay-gate bottleneck (d_head, d_model): d_head is held fixed,
        # so only the input is widened.
        return _in(W, r, noise, gen)
    if k.endswith(".f_proj.1.weight"):
        # GDN-2 decay-gate expansion (key_dim, d_head): the bottleneck input is
        # unchanged, the output tiles per copied head (copy-major, matching the
        # widened proj_k layout). Noise-free so the r copies stay exact.
        return W.repeat(r, 1)
    if k.endswith(".dt_bias") or k.endswith(".A_log"):
        # per key channel (dt_bias) / per key head (A_log) -> replicate per
        # copied head (copy-major, matching the widened proj_k layout)
        return W.repeat(r)
    if k.endswith(".norm.weight"):
        return W.clone()  # per-d_head (d_head unchanged)
    if k.endswith(".out_gain"):
        return W.repeat(r)  # MoE output RMSNorm gain, per (copied) d_model channel
    if ".mtp_heads." in k:
        if k.endswith(".proj_in.weight"):
            return _in(W, r, noise, gen)  # (rank, d_model): input-widened only
        # out.weight: (d_model, d_model) when full-rank -> both-widened; else
        # (d_model, rank) -> output-widened only (tile the rows). Kept noise-free
        # so a zero-initialized MTP head stays exactly the identity at init.
        if src["mtp_rank"] is None:
            return _both(W, r, 0.0, gen)
        return W.repeat(r, 1)
    # Everything else is a 2D linear with both dims d-widened (all attention/FFN
    # projections, the channel-wise b/w gates, the NSA branch gate).
    return _both(W, r, noise, gen)


def grow_width(
    state: dict[str, Tensor],
    src: dict,
    tgt: dict,
    noise: float = 0.01,
    seed: int = 0,
) -> dict[str, Tensor]:
    """Widen `state` (a bare dense TransformerLM state_dict) from `src` to `tgt`
    config by an integer factor r = tgt.d_model / src.d_model, via HyperCloning.
    d_head must be equal in both, and every width field (n_heads, n_kv_heads,
    nsa_kv_heads, d_ffn) must scale by exactly r. Returns a new state_dict with
    the same keys and r-widened tensors; the built model computes the identical
    function at init (up to floating point)."""
    assert src["n_experts"] is None, "width growth of a MoE model is unsupported"
    dm_s, dm_t = src["d_model"], tgt["d_model"]
    assert dm_t > dm_s and dm_t % dm_s == 0, "d_model must grow by an integer factor"
    r = dm_t // dm_s
    assert src["d_head"] == tgt["d_head"], (
        f"width growth needs constant d_head (src {src['d_head']} != tgt "
        f"{tgt['d_head']}); widen the head count, not the head size"
    )
    for f in ("n_heads", "n_kv_heads", "nsa_kv_heads", "d_ffn"):
        assert tgt[f] == r * src[f], f"{f} must scale by {r} for width growth"
    gen = torch.Generator().manual_seed(seed)
    return {k: _widen_key(k, W, r, src, noise, gen) for k, W in state.items()}


# ---------------------------------------------------------------------------
# depth (block stacking)
# ---------------------------------------------------------------------------


def _layer_prefix(i: int) -> str:
    return f"transformer.layers.{i}."


def _copy_layer(
    state: dict[str, Tensor],
    out: dict[str, Tensor],
    src_i: int,
    dst_i: int,
    zero_residual: bool,
) -> None:
    """Copy all tensors of layer `src_i` into layer `dst_i`. With zero_residual,
    the mixer/FFN/expert output projections are zeroed instead of copied, so the
    destination layer contributes nothing to the residual at init."""
    pre = _layer_prefix(src_i)
    for k, W in state.items():
        if not k.startswith(pre):
            continue
        tail = k[len(pre) :]
        nk = _layer_prefix(dst_i) + tail
        out[nk] = (
            torch.zeros_like(W)
            if (zero_residual and tail in _RESIDUAL_WRITE)
            else W.clone()
        )


def grow_depth(
    state: dict[str, Tensor], src: dict, tgt: dict, seed: int = 0
) -> dict[str, Tensor]:
    """Deepen `state` from src.n_layers to tgt.n_layers by appending whole
    blocks. Existing layers are copied unchanged; each appended layer is seeded
    from the same-role layer of the last source block (so it inherits trained
    features) but has its residual-writing projections zeroed, starting as a
    near-identity.

    Only *whole blocks* are appended (the growth must be a multiple of
    layers_per_block) so the GDN:NSA role pattern and the block-boundary
    structure are preserved. Approximately -- not exactly -- function-preserving:
    a zeroed block still commits a zero block representation, and picochat's
    DepthAttention residual (a softmax over all block representations, see
    gpt.DepthAttention) renormalizes over that extra entry, so downstream mixing
    weights shift slightly. The perturbation is small and training absorbs it;
    the backbone (embeddings, every existing layer, the head) is carried over
    bit-for-bit."""
    lpb = tgt["layers_per_block"]
    L_s, L_t = src["n_layers"], tgt["n_layers"]
    assert lpb == src["layers_per_block"], "layers_per_block must match"
    assert L_t > L_s, "target must be deeper"
    assert L_s % lpb == 0 and L_t % lpb == 0, "layer counts must be whole blocks"
    for f in ("d_model", "n_heads", "n_kv_heads", "nsa_kv_heads", "d_ffn", "n_experts"):
        assert src[f] == tgt[f], f"grow_depth cannot change {f}"
    out = {
        k: W.clone()
        for k, W in state.items()
        if not k.startswith("transformer.layers.")
    }
    for i in range(L_s):
        _copy_layer(state, out, i, i, zero_residual=False)
    for i in range(L_s, L_t):
        # seed from the matching position in the last source block (same role:
        # linear GDN vs. NSA tail is decided by i % lpb, preserved here)
        src_i = (L_s - lpb) + (i % lpb)
        _copy_layer(state, out, src_i, i, zero_residual=True)
    return out


# ---------------------------------------------------------------------------
# dense -> MoE (sparse upcycling)
# ---------------------------------------------------------------------------


def _fresh(shape, std: float, gen: torch.Generator) -> Tensor:
    return torch.randn(*shape, generator=gen) * std


def upcycle_to_moe(
    state: dict[str, Tensor], src: dict, tgt: dict, seed: int = 0
) -> dict[str, Tensor]:
    """Add a routed MoE branch to every layer of a dense `state`, turning it into
    the `tgt` MoE model. The routed experts' output projection is zeroed
    (weight_expand for a LatentMoE, else the experts' weight_down), so the MoE
    contributes nothing at init and the model starts as the dense one. Experts'
    input/hidden weights and the router are freshly initialized; training grows
    the experts off zero and specializes them (the router still runs at init, it
    just has no effect on the output).

    Function-preserving iff the dense/shared FFN width is preserved (tgt.d_ffn ==
    src.d_ffn): that dense FFN is kept as the always-on shared expert. When the
    target uses a smaller shared FFN (the usual MoE-rung design, to keep the
    active-parameter count low), the FFN is reshaped and re-initialized instead,
    so the init is a warm start (backbone carried over) rather than exactly
    function-preserving; the caller is warned."""
    assert src["n_experts"] is None, "source model must be dense"
    assert tgt["n_experts"] is not None, "target must be a MoE model"
    for f in ("d_model", "n_heads", "n_kv_heads", "nsa_kv_heads", "n_layers"):
        assert src[f] == tgt[f], f"upcycle cannot change {f}"
    gen = torch.Generator().manual_seed(seed)
    d_model, n_layers, std = tgt["d_model"], tgt["n_layers"], tgt["init_std"]
    ne, d_expert, d_latent = tgt["n_experts"], tgt["d_expert"], tgt["d_latent"]
    shared, latent = tgt["share_experts"], d_latent is not None
    io = d_latent if latent else d_model
    ffn_preserved = tgt["d_ffn"] == src["d_ffn"]

    out = {k: W.clone() for k, W in state.items()}

    # One shared ExpertBank for the whole stack (MoEUT-style) vs. an independent
    # bank per layer. A shared bank appears in the state_dict under every layer's
    # keys (aliased storage); writing the same tensors to each keeps them consistent.
    def make_bank() -> dict[str, Tensor]:
        wd = (
            torch.zeros(ne * io, d_expert)
            if not latent
            else _fresh((ne * io, d_expert), std, gen)
        )
        return {
            "bank.weight_up": _fresh((ne * d_expert, io), std, gen),
            "bank.weight_gate": _fresh((ne * d_expert, io), std, gen),
            "bank.weight_down": wd,  # zeroed here when non-latent: the residual write
        }

    shared_bank = make_bank() if shared else None
    for i in range(n_layers):
        p = _layer_prefix(i) + "moe."
        out[p + "weight_router"] = _fresh((ne, d_model), std, gen)
        out[p + "out_gain"] = torch.ones(d_model)
        out[p + "expert_bias"] = torch.zeros(ne)
        if latent:
            out[p + "weight_compress"] = _fresh((d_latent, d_model), std, gen)
            out[p + "weight_expand"] = torch.zeros(d_model, d_latent)  # residual write
        bank = shared_bank if shared else make_bank()
        for name, W in bank.items():
            out[p + name] = W
        if not ffn_preserved:
            _reinit_ffn(out, i, d_model, tgt["d_ffn"], n_layers, std, gen)
    return out


def _reinit_ffn(out, i, d_model, d_ffn, n_layers, std, gen) -> None:
    p = _layer_prefix(i) + "ffn."
    out[p + "proj_up.weight"] = _fresh((d_ffn, d_model), std, gen)
    out[p + "proj_gate.weight"] = _fresh((d_ffn, d_model), std, gen)
    # residual-write projection: the depth-scaled std TransformerLM._init_weights uses
    out[p + "proj_down.weight"] = _fresh(
        (d_model, d_ffn), std / (2 * n_layers) ** 0.5, gen
    )


# ---------------------------------------------------------------------------
# multi-token-prediction head count
# ---------------------------------------------------------------------------


def _adjust_mtp(state: dict[str, Tensor], src: dict, tgt: dict) -> dict[str, Tensor]:
    """Match the MTP head count to the target. Extra heads are dropped; new heads
    are added zero-initialized (MTPHead is the identity at init -- its residual
    transform outputs zero -- so adding one is function-preserving)."""
    assert src["mtp_rank"] == tgt["mtp_rank"], "changing mtp_rank is unsupported"
    n_s, n_t, d_model, rank = (
        src["n_mtp"],
        tgt["n_mtp"],
        tgt["d_model"],
        tgt["mtp_rank"],
    )
    out = {k: v for k, v in state.items() if not k.startswith("mtp_heads.")}
    for j in range(min(n_s, n_t)):  # carry over existing heads
        for suf in ["proj_in.weight", "out.weight"] if rank else ["out.weight"]:
            out[f"mtp_heads.{j}.{suf}"] = state[f"mtp_heads.{j}.{suf}"].clone()
    for j in range(n_s, n_t):  # fresh identity heads
        out[f"mtp_heads.{j}.out.weight"] = torch.zeros(
            d_model, d_model if rank is None else rank
        )
        if rank is not None:
            out[f"mtp_heads.{j}.proj_in.weight"] = (
                torch.randn(rank, d_model) * tgt["init_std"]
            )
    return out


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def grow_state_dict(
    state: dict[str, Tensor],
    source_cfg: dict,
    target_cfg: dict,
    noise: float = 0.01,
    seed: int = 0,
) -> dict[str, Tensor]:
    """Grow a bare TransformerLM state_dict from `source_cfg` to `target_cfg`,
    applying width -> depth -> (MTP heads) -> MoE-upcycle as needed. Both configs
    are preset/model_config dicts (defaults are normalized internally). Returns a
    state_dict whose keys and shapes match build_lm(**target_cfg); load it with
    strict=True. See the module docstring for which stages are exactly vs.
    approximately function-preserving."""
    src, tgt = _norm_cfg(source_cfg), _norm_cfg(target_cfg)
    assert src["vocab_size"] == tgt["vocab_size"], "growth cannot change the vocab"
    s = state

    if tgt["d_model"] != src["d_model"]:
        r = tgt["d_model"] // src["d_model"]
        wide = {**src, "d_model": tgt["d_model"], "d_head": tgt["d_head"]}
        for f in ("n_heads", "n_kv_heads", "nsa_kv_heads", "d_ffn"):
            wide[f] = r * src[f]  # grow_width asserts these match the true target
        s = grow_width(s, src, wide, noise=noise, seed=seed)
        src = wide

    if tgt["n_layers"] != src["n_layers"]:
        deep = {**src, "n_layers": tgt["n_layers"]}
        s = grow_depth(s, src, deep, seed=seed)
        src = deep

    if tgt["n_mtp"] != src["n_mtp"]:
        s = _adjust_mtp(s, src, tgt)
        src = {**src, "n_mtp": tgt["n_mtp"]}

    if tgt["n_experts"] is not None and src["n_experts"] is None:
        s = upcycle_to_moe(s, src, tgt, seed=seed)
        src = {**src, "n_experts": tgt["n_experts"]}
    elif tgt["n_experts"] != src["n_experts"]:
        raise ValueError(
            "growing between two MoE configs (changing the expert pool) is "
            "unsupported; grow from a dense source"
        )

    return s
