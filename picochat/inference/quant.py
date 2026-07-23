"""Weight-only int8 quantization for inference.

Symmetric per-output-channel int8: each row of an nn.Linear weight is scaled
into [-127, 127] by its own absmax and stored as int8 plus one fp16 scale.
The matmul itself stays dense floating point -- forward dequantizes the weight
to the activation dtype and calls F.linear -- so this is purely a *memory*
optimization (4x smaller than fp32 weights, 2x smaller than bf16), which is
what matters for fitting a checkpoint on small CPU/GPU boxes; per-channel
scaling keeps the round-trip error well under 1% relative, small enough that
greedy decoding is essentially unchanged.

Scope and limitations:
- Only nn.Linear submodules are converted. The MoE ExpertBank stores its
  routed-expert weights as raw nn.Parameter matrices (picochat.model.moe),
  not nn.Linear, so MoE presets keep their expert weights in full precision;
  the dense presets are fully covered.
- The token embedding and the lm head are skipped by default (the standard
  recipe: both touch the vocab-sized matrix whose quantization error shows up
  directly in the logits).
- Bias is unsupported; the picochat model uses bias=False everywhere.

Entry point: load_gpt_checkpoint(..., dtype="int8") in
picochat.training.checkpoint, exposed as --dtype int8 in scripts/chat.py and
scripts/api.py. Activations stay fp32 (no autocast) -- do not combine with
bf16/fp16 weight casts.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Int8Linear(nn.Module):
    """Drop-in replacement for a bias-free nn.Linear with a weight-only
    symmetric per-output-channel int8 weight.

    State is stored in *buffers* (not parameters): `weight_q` (out, in) int8
    and `scale` (out, 1) fp16, so the module is inference-only by construction
    (nothing to train) and `inference_autocast` -- which keys on half-precision
    *parameters* -- stays a no-op for an int8-quantized model."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_q", torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        # fp16 scales: the per-row absmax/127 values are tiny but well within
        # fp16 range, and fp16's ~1e-3 relative precision is negligible next
        # to int8's ~0.4% rounding error -- so spend 2 bytes/row, not 4.
        self.register_buffer("scale", torch.ones(out_features, 1, dtype=torch.float16))

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        """Quantize an existing nn.Linear (bias unsupported -- the picochat
        model is bias-free everywhere)."""
        assert linear.bias is None, "Int8Linear does not support bias"
        w = linear.weight.detach().float()
        # Symmetric per-output-channel scaling: each row maps its absmax onto
        # 127. Clamp guards the division for an all-zero row (its q ends up 0
        # anyway, so the stored scale value is irrelevant there).
        scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-12)
        q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
        mod = cls(linear.in_features, linear.out_features)
        # assign (rather than copy_) so the buffers land on the source
        # weight's device -- quantization may run after model.to(device)
        mod.weight_q = q.contiguous()
        mod.scale = scale.to(torch.float16)
        return mod

    def forward(self, x: Tensor) -> Tensor:
        # Dequantize to the *activation* dtype: exact under fp32, and under a
        # bf16/fp16 autocast region the weight materializes directly in the
        # low precision instead of round-tripping through fp32.
        w = self.weight_q.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


def quantize_model_int8(
    model: nn.Module, skip: tuple[str, ...] = ("embed", "lmhead")
) -> nn.Module:
    """Swap every nn.Linear submodule of `model` for an Int8Linear built from
    its weights, in place, and return the model.

    Modules whose qualified name contains any `skip` substring are left in
    full precision (default: the token embedding and the lm head -- see the
    module docstring), as are modules that are already Int8Linear (repeat
    calls are no-ops). ExpertBank parameters are untouched (raw nn.Parameter,
    not nn.Linear)."""
    # Collect first, replace after: mutating the module tree while
    # named_modules() is iterating over it is undefined behavior.
    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            qualified = f"{name}.{child_name}" if name else child_name
            if not isinstance(child, nn.Linear):
                continue  # also skips already-converted Int8Linear modules
            if any(s in qualified for s in skip):
                continue
            targets.append((module, child_name, child))
    for parent, child_name, child in targets:
        # setattr works for nn.Sequential's numeric child names too --
        # nn.Module.__setattr__ routes Modules into parent._modules.
        setattr(parent, child_name, Int8Linear.from_linear(child))
    print(f"quantized {len(targets)} Linear layers to weight-only int8", flush=True)
    return model
