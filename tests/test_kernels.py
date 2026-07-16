"""picochat.kernels: the optional HF-Hub kernel integration. fused_loss is
opt-in; without CUDA (or the `kernels` package) it must fail loudly rather
than silently training differently, and when the kernel is genuinely usable
the fused loss must match the plain one."""

import copy

import pytest
import torch

from picochat.gpt import TransformerLM
from picochat.kernels import fused_linear_cross_entropy_available
from picochat.trainer import GPT, SFTModule


def _tiny_lm(vocab_size: int = 40) -> TransformerLM:
    return TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2)


def test_fused_loss_defaults_off_and_uses_full_model():
    for module in (GPT(_tiny_lm(), pad_idx=0), SFTModule(_tiny_lm(), pad_idx=0)):
        assert module.fused_loss is False
        # unfused: the forward handle wraps the full model (logits), not the
        # encode trunk; compare unwrapped in case torch.compile is active
        forward = getattr(module._forward, "_orig_mod", module._forward)
        assert forward is module.model


@pytest.mark.skipif(
    fused_linear_cross_entropy_available(),
    reason="kernel available here; the failure path needs it absent",
)
def test_fused_loss_true_raises_when_unavailable():
    with pytest.raises(RuntimeError, match="fused_loss"):
        GPT(_tiny_lm(), pad_idx=0, fused_loss=True)
    with pytest.raises(RuntimeError, match="fused_loss"):
        SFTModule(_tiny_lm(), pad_idx=0, fused_loss=True)


@pytest.mark.skipif(
    not fused_linear_cross_entropy_available(),
    reason="needs CUDA + the kernels extra (fetches liger-kernels from the Hub)",
)
def test_fused_loss_matches_plain_loss_on_cuda():
    torch.manual_seed(0)
    lm = TransformerLM(
        vocab_size=128, d_model=64, n_heads=4, n_layers=2, grad_checkpoint=False
    )
    plain = GPT(lm, pad_idx=0, bos_idx=1, compile=False, fused_loss=False)
    fused = GPT(copy.deepcopy(lm), pad_idx=0, bos_idx=1, compile=False, fused_loss=True)
    # eval(): dropout would otherwise draw different masks per call
    plain, fused = plain.cuda().eval(), fused.cuda().eval()

    x = torch.randint(2, 128, (4, 32), device="cuda")
    x[:, 0] = 1  # bos
    x[:, -3:] = 0  # pad tail (ignored targets)
    loss_plain, loss_fused = plain._loss(x), fused._loss(x)
    assert torch.allclose(loss_plain, loss_fused, atol=1e-4)

    # the kernel computes gradients internally -- they must match autograd's
    loss_plain.backward()
    loss_fused.backward()
    for (name, p1), (_, p2) in zip(
        plain.model.named_parameters(), fused.model.named_parameters()
    ):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4), name
