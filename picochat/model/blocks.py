"""Model-independent building blocks: the normalization/FFN primitives, the
depth-attention residual mixing, and the sequence-packing boundary helper.
The sequence mixers live in linear_attn.py / sparse_attn.py, the MoE in
moe.py, and everything is assembled in transformer.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def rms_norm(x: Tensor, eps: float = 1e-8) -> Tensor:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        return x / (x.square().mean(-1, keepdim=True).sqrt() + eps)


def doc_ids_to_cu_seqlens(doc_ids: Tensor) -> Tensor:
    """Segment boundaries (into the flattened batch*seq layout) at which a Gated
    DeltaNet-2 layer must reset its recurrent state: every document boundary
    within a packed row, plus every row boundary (so flattening rows never lets
    the recurrence leak across them). Returns a 1D LongTensor starting at 0 and
    ending at batch*seq. Data-dependent, so built outside the compiled forward
    (like the old packed_masks)."""
    b, seq = doc_ids.shape
    flat = doc_ids.clone()
    # make doc ids unique per row so a row boundary always reads as a change
    flat = flat + (
        torch.arange(b, device=doc_ids.device)[:, None] * (doc_ids.max() + 1)
    )
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
        # picochat.training.optim.muon_param_split).
        self.query = nn.Parameter(torch.zeros(d_model))

    def forward(self, blocks: Tensor, partial: Tensor | None) -> Tensor:
        # blocks (n, b, t, d): committed block representations, blocks[0] being
        # the token embedding. partial (b, t, d): the current block's running
        # sum of sublayer outputs; None at a block's first sublayer, where only
        # completed blocks are visible.
        values = blocks if partial is None else torch.cat([blocks, partial[None]])
        weight = (rms_norm(values) * self.query).sum(-1).softmax(0)
        return (values * weight[..., None]).sum(0)
