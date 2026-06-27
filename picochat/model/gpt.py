import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def rms_norm(x: Tensor, eps: float = 1e-8) -> Tensor:
    return x / (x.square().mean(-1, keepdim=True).sqrt() + eps)


def rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = torch.stack([-x2, x1], dim=-1)
    x = rearrange(x, "... d r -> ... (d r)")
    return x


class SwiGLU(nn.Module):
    pass


class CausalSelfAttention(nn.Module):
    pass
