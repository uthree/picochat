"""Optional Hugging Face `kernels` integration: hub-loaded Triton kernels
with a plain-PyTorch fallback (the picochat analogue of nanochat's
flash_attention.py wrapper).

The only kernel used today is Liger's fused linear cross-entropy
(kernels-community/liger-kernels): it folds the lm-head matmul into a chunked
cross-entropy so the (b*l, vocab) logits tensor is never materialized. With a
128k vocab the logits are by far the largest activation of a training step
(batch 8 x 2048 tokens x 128k vocab in bf16 is ~4 GiB *before* the fp32
softmax the plain loss needs), so the fused path frees enough memory for a
several-times-larger batch at the same model size. Attention and the other
blocks stay on FlexAttention/torch.compile, which already fuse well and --
unlike the loss -- have no memory cliff.

The integration is opt-in (`trainer.fused_loss: true` in a stage config): the
chunked kernel trades some step time for the memory on smaller GPUs, so it is
a deliberate choice when memory-bound rather than a silent default. The
`kernels` package is an optional dependency (`pip install picochat[kernels]`)
and the kernel itself is fetched from the Hub on first use (then cached by
huggingface_hub); anything missing -- no CUDA, no package, no network and no
cache -- resolves to "not available", which the trainers turn into a loud
error instead of silently training slower than configured. Tests and CPU
runs never touch the Hub.
"""

import functools
import warnings

import torch
from torch import Tensor

LIGER_REPO = "kernels-community/liger-kernels"
# Version branch, pinned: the kernels library guarantees API stability within
# a version branch, not across them.
LIGER_VERSION = 2


@functools.cache
def _liger():
    """The loaded liger-kernels module, or None if unavailable (no CUDA, no
    `kernels` package, or the Hub fetch failed). Cached: one attempt per
    process, so an offline machine warns once instead of retrying per step."""
    if not torch.cuda.is_available():
        return None
    try:
        from kernels import get_kernel
    except ImportError:
        return None
    try:
        return get_kernel(LIGER_REPO, version=LIGER_VERSION)
    except Exception as e:  # offline and uncached, hub outage, ...
        warnings.warn(
            f"could not load {LIGER_REPO} (v{LIGER_VERSION}) from the HF Hub "
            f"({type(e).__name__}: {e}); falling back to the plain loss",
            stacklevel=2,
        )
        return None


def fused_linear_cross_entropy_available() -> bool:
    """Whether fused_linear_cross_entropy can run here (CUDA + `kernels`
    package + the liger kernel loadable). Probing may hit the Hub once."""
    return _liger() is not None


def fused_linear_cross_entropy(
    hidden: Tensor, weight: Tensor, targets: Tensor, ignore_index: int
) -> Tensor:
    """Mean cross-entropy of `hidden @ weight.T` against `targets`, without
    materializing the logits (Liger chunks the matmul and fuses the softmax
    into it; backward is fully supported, with the weight-gradient
    accumulated in fp32 for bf16 stability).

    hidden: (n, d) -- flatten (b, l, d) first; targets: (n,) int64. The
    weight is cast to hidden's dtype exactly like autocast would for the
    F.linear it replaces (the cast is differentiable, so fp32 master weights
    under bf16-mixed still receive their gradient). Callers must check
    fused_linear_cross_entropy_available() first.
    """
    return _liger().layers.liger_fused_linear_cross_entropy(
        hidden.contiguous(),
        weight.to(hidden.dtype),
        targets.contiguous(),
        ignore_index=ignore_index,
        reduction="mean",
        accum_dtype=torch.float32,
    )
