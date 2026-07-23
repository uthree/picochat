"""Optimizer wiring and the LR schedule, as plain functions over a model:
which parameters Muon may orthogonalize vs. which stay on AdamW, the
decay/no-decay split, and the warmup+cosine schedule. The LightningModules in
modules.py delegate here (they own *when* to step; this module owns *what*
gets which treatment).

The "muon" mode runs two optimizers side by side: torch.optim.Muon for the
matrix-shaped hidden weights and torch.optim.AdamW for the rest (embeddings,
lm head, 1-dim params) -- torch's Muon is Muon-only, unlike the previous
in-repo implementation that embedded its own AdamW.
"""

import math

import torch.nn as nn

from picochat.model.sparse_attn import NativeSparseAttention


def _embedding_param_ids(model: nn.Module) -> set[int]:
    """ids of the embedding parameters -- excluded from weight decay and,
    under Muon, routed to AdamW like the input/output layers."""
    return {
        id(p)
        for m in model.modules()
        if isinstance(m, nn.Embedding)
        for p in m.parameters()
    }


def _non_muon_param_ids(model: nn.Module) -> set[int]:
    # Params that must NOT go to Muon even though they are 2D. Native Sparse
    # Attention's positional signal today is PartialRoPE -- non-persistent
    # sin/cos buffers, not parameters -- so NSA contributes nothing here. The
    # getattr guard keeps this correct if a *learned* positional table
    # (`cmp_pos`) is reintroduced: such a table is a lookup, not a hidden
    # linear map, so it belongs on AdamW (no-decay) like the embeddings.
    # (Depthwise conv weights are 3D and are excluded by the ndim check in the
    # callers; Muon accepts only 2D matrices.)
    ids = set()
    for m in model.modules():
        if isinstance(m, NativeSparseAttention):
            pos = getattr(m, "cmp_pos", None)
            if pos is not None:
                ids.add(id(pos))
    return ids


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """AdamW-only parameter groups: weight decay for 2+-dim weights, none for
    biases (1-dim) and embeddings (rms_norm has no learnable params, so
    nothing to exclude there)."""
    embed_ids = _embedding_param_ids(model)
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or id(p) in embed_ids:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def muon_param_split(model: nn.Module, weight_decay: float) -> tuple[list, list[dict]]:
    """(muon_params, adamw_groups) for the side-by-side Muon+AdamW mode.

    Muon orthogonalizes matrix-shaped *hidden* weights. The embedding and
    lm heads (input/output layers, per the Muon authors), 1-dim params
    (biases, norms, gates, A_log/dt_bias), 3-dim depthwise conv weights,
    and the NSA positional table go to the AdamW running alongside it
    instead, keeping the same decay split as param_groups: no decay for
    embeddings/1-dim/positional, decay for lm-head and conv matrices.
    Everything else -- attention/FFN/mixer projections, the router, the
    NSA gate/compression MLPs, and the fused MoE expert weights (stored 2D
    exactly because torch.optim.Muon accepts nothing else) -- is Muon's."""
    embed_ids = _embedding_param_ids(model)
    no_decay_ids = embed_ids | _non_muon_param_ids(model)
    # The lm head is the model's output projection -> AdamW, not Muon (Muon
    # skips input/output layers). The MTP heads' transforms are hidden
    # d_model x d_model matrices (they reuse this same output projection), so
    # they go to Muon like every other hidden weight.
    head_ids = {id(p) for p in model.lmhead.parameters()}
    muon, adam_decay, adam_no_decay = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or id(p) in no_decay_ids:
            adam_no_decay.append(p)
        elif id(p) in head_ids or p.ndim != 2:
            # lm head, and any >2D weight (depthwise conv) Muon can't take
            adam_decay.append(p)
        else:
            muon.append(p)
    return muon, [
        dict(params=adam_decay, weight_decay=weight_decay),
        dict(params=adam_no_decay, weight_decay=0.0),
    ]


def lr_lambda(
    step: int, warmup_steps: int, max_steps: int | None, min_lr_ratio: float
) -> float:
    """Linear warmup -> cosine decay (down to min_lr_ratio), as a base-LR
    multiplier."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    if max_steps is None or step >= max_steps:
        return min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * coeff
