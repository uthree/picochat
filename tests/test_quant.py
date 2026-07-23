"""Weight-only int8 quantization (picochat.inference.quant): round-trip
accuracy on plain Linears, model surgery on a tiny TransformerLM, and the
end-to-end quantized decode path through the inference engine."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from picochat.inference.engine import SamplingConfig, generate
from picochat.inference.quant import Int8Linear, quantize_model_int8
from picochat.model import TransformerLM
from picochat.tokenizer import SPECIAL_TOKENS


class ByteTokenizer:
    """1 byte = 1 token; special tokens get ids >= 256 (same stub as
    tests/test_generate.py). With vocab < 256 in the tiny models below the
    stop-token ids are unreachable, so generation always runs to its budget."""

    def __init__(self):
        self._special = {tok: 256 + i for i, tok in enumerate(SPECIAL_TOKENS)}

    def encode_ordinary(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_single_token(self, token: str) -> int:
        return self._special[token]

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return bytes([token_id]) if token_id < 256 else b"?"

    def decode(self, ids: list[int]) -> str:
        joined = b"".join(self.decode_single_token_bytes(i) for i in ids)
        return joined.decode("utf-8", errors="replace")


def _tiny_lm(vocab_size=128) -> TransformerLM:
    torch.manual_seed(0)
    return TransformerLM(
        vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2
    ).eval()


# ---------------------------------------------------------------------------
# Int8Linear
# ---------------------------------------------------------------------------
def test_dequantized_weight_error_is_small():
    torch.manual_seed(0)
    linear = nn.Linear(128, 256, bias=False)
    q = Int8Linear.from_linear(linear)
    dequant = q.weight_q.float() * q.scale.float()
    rel = (dequant - linear.weight).norm() / linear.weight.norm()
    # per-channel symmetric int8 sits well under 1% relative error
    assert rel.item() < 0.01


def test_output_close_to_fp32():
    torch.manual_seed(0)
    linear = nn.Linear(64, 96, bias=False)
    q = Int8Linear.from_linear(linear)
    x = torch.randn(8, 64)
    ref = linear(x)
    out = q(x)
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    assert cos.item() > 0.999


def test_forward_follows_input_dtype():
    torch.manual_seed(0)
    q = Int8Linear.from_linear(nn.Linear(16, 16, bias=False))
    # autocast-friendly: the weight dequantizes to the activation dtype
    assert q(torch.randn(2, 16, dtype=torch.bfloat16)).dtype == torch.bfloat16
    assert q(torch.randn(2, 16)).dtype == torch.float32


def test_from_linear_rejects_bias():
    with pytest.raises(AssertionError):
        Int8Linear.from_linear(nn.Linear(8, 8, bias=True))


def test_zero_row_quantizes_to_zero():
    linear = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        linear.weight[1].zero_()
    q = Int8Linear.from_linear(linear)
    assert (q.weight_q[1] == 0).all()
    assert torch.isfinite(q(torch.randn(2, 8))).all()


def test_int8_buffers_use_under_half_the_memory():
    linear = nn.Linear(256, 512, bias=False)
    q = Int8Linear.from_linear(linear)
    q_bytes = sum(b.numel() * b.element_size() for b in q.buffers())
    w_bytes = linear.weight.numel() * linear.weight.element_size()
    assert q_bytes < w_bytes / 2


# ---------------------------------------------------------------------------
# quantize_model_int8
# ---------------------------------------------------------------------------
def test_quantize_model_swaps_projections_but_not_embed_lmhead():
    lm = _tiny_lm()
    quantize_model_int8(lm)
    # every mixer/FFN projection got swapped -- only the lm head remains a
    # plain nn.Linear (the embedding is an nn.Embedding, untouched by design)
    remaining = [n for n, m in lm.named_modules() if isinstance(m, nn.Linear)]
    assert remaining == ["lmhead"]
    assert isinstance(lm.embed, nn.Embedding)
    swapped = [n for n, m in lm.named_modules() if isinstance(m, Int8Linear)]
    assert len(swapped) > 0
    assert not any("lmhead" in n or "embed" in n for n in swapped)


def test_quantize_model_is_idempotent():
    lm = _tiny_lm()
    quantize_model_int8(lm)
    before = [n for n, m in lm.named_modules() if isinstance(m, Int8Linear)]
    quantize_model_int8(lm)  # second pass must be a no-op, not a re-quantize
    after = [n for n, m in lm.named_modules() if isinstance(m, Int8Linear)]
    assert before == after


def test_quantized_logits_close_to_fp32():
    lm = _tiny_lm()
    torch.manual_seed(1)
    x = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        ref, _, _ = lm.decode(x)
    quantize_model_int8(lm)
    with torch.no_grad():
        out, _, _ = lm.decode(x)
    # per-position logits cosine similarity, plus greedy top-1 agreement
    cos = F.cosine_similarity(out[0], ref[0], dim=-1)
    assert cos.min().item() > 0.99
    agree = (out[0].argmax(-1) == ref[0].argmax(-1)).float().mean()
    assert agree.item() >= 0.9


def test_quantized_model_generates():
    lm = _tiny_lm()
    quantize_model_int8(lm)
    tok = ByteTokenizer()
    cfg = SamplingConfig(temperature=0.0, max_new_tokens=6)
    out = list(generate(lm, tok, [1, 2, 3, 4], cfg, max_seq_len=64))
    # vocab 128 keeps stop ids (>= 256) unreachable: the full budget streams
    assert len(out) == 6
    assert all(isinstance(t, int) and 0 <= t < 128 for t in out)
