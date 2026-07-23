"""Multi-token prediction (MTP) heads and self-speculative decoding.

The correctness contract: generate_speculative must emit exactly the greedy
(argmax) token stream, because it verifies every drafted token against the
model's real next-token prediction and only accepts matches. So it is tested
against a plain one-token-at-a-time greedy reference -- across full-attention,
sliding-window and MoE configs, so the KV-cache rollback after rejected drafts
is exercised in each."""

import pytest
import torch

from picochat.inference.engine import SamplingConfig, generate, generate_speculative
from picochat.model import TransformerLM
from picochat.tokenizer import (
    EOS_TOKEN,
    IM_END,
    SPECIAL_TOKENS,
    load_tokenizer,
    train_tokenizer,
)


def _tokenizer(tmp_path):
    corpus = ["the quick brown fox jumps over the lazy dog near the river"] * 60
    path = tmp_path / "tok.json"
    train_tokenizer(
        iter(corpus), vocab_size=320, save_as=path, special_tokens=SPECIAL_TOKENS
    )
    return load_tokenizer(path)


def _greedy_reference(model, prompt_ids, budget, stop_ids):
    """Plain one-token-at-a-time argmax decode -- the ground truth stream."""
    x = torch.tensor([prompt_ids])
    logits, cache, pos = model.decode(x)
    tok = int(logits[0, -1].argmax())
    out = []
    for _ in range(budget):
        if tok in stop_ids:
            break
        out.append(tok)
        logits, cache, pos = model.decode(torch.tensor([[tok]]), cache, pos)
        tok = int(logits[0, -1].argmax())
    return out


def _model(vocab, n_mtp, **kw):
    kw.setdefault("d_model", 32)
    kw.setdefault("n_heads", 4)
    kw.setdefault("n_layers", 4)
    return TransformerLM(vocab_size=vocab, max_seq_len=128, n_mtp=n_mtp, **kw).eval()


def test_decode_heads_shapes():
    torch.manual_seed(0)
    lm = _model(50, n_mtp=2)
    x = torch.randint(0, 50, (1, 6))
    logits, mtp, cache, pos = lm.decode_heads(x)
    assert logits.shape == (1, 6, 50)
    assert len(mtp) == 2 and all(m.shape == (1, 6, 50) for m in mtp)
    assert pos == 6


def test_mtp_heads_start_as_identity():
    # zero-initialized transforms -> at init every MTP head predicts exactly like
    # the shared primary head (a sensible starting point before specialization).
    torch.manual_seed(0)
    lm = _model(50, n_mtp=2)
    logits, mtp, _, _ = lm.decode_heads(torch.randint(0, 50, (1, 4)))
    for m in mtp:
        assert torch.allclose(m, logits, atol=1e-5)


def test_mtp_low_rank_head_is_cheaper():
    # low-rank heads factor the d_model transform through a bottleneck.
    full = _model(50, n_mtp=2)
    low = _model(50, n_mtp=2, mtp_rank=4)

    def n(lm):
        return sum(p.numel() for h in lm.mtp_heads for p in h.parameters())

    assert n(low) < n(full)


@pytest.mark.parametrize(
    "kw",
    [
        dict(n_layers=4, layers_per_block=1),  # all full attention, own blocks
        dict(n_layers=4, layers_per_block=2, window_size=4),  # windowed + blocks
        dict(
            n_layers=4, n_experts=6, n_active=2, d_expert=16, layers_per_block=1
        ),  # MoE
        dict(n_layers=4, layers_per_block=1, mtp_rank=8),  # low-rank MTP heads
    ],
)
@pytest.mark.parametrize("n_mtp", [1, 3])
def test_speculative_matches_greedy(tmp_path, kw, n_mtp):
    # identical output to plain greedy decoding, regardless of how many drafts
    # are accepted or rejected each step (exercises the cache rollback).
    torch.manual_seed(0)
    tok = _tokenizer(tmp_path)
    model = _model(tok.n_vocab, n_mtp=n_mtp, **kw)
    prompt = tok.encode_ordinary("the quick brown fox")
    stop = {tok.encode_single_token(IM_END), tok.encode_single_token(EOS_TOKEN)}

    ref = _greedy_reference(model, prompt, budget=25, stop_ids=stop)
    cfg = SamplingConfig(max_new_tokens=25)
    spec = list(generate_speculative(model, tok, prompt, cfg, max_seq_len=128))
    assert spec == ref


def test_speculative_respects_max_seq_len(tmp_path):
    # the verify chunk must never push RoPE past max_seq_len.
    torch.manual_seed(0)
    tok = _tokenizer(tmp_path)
    model = _model(tok.n_vocab, n_mtp=3)
    prompt = tok.encode_ordinary("the quick brown")
    cfg = SamplingConfig(max_new_tokens=100)
    out = list(generate_speculative(model, tok, prompt, cfg, max_seq_len=20))
    assert len(prompt) + len(out) <= 20


def test_generate_routes_greedy_to_speculative(tmp_path):
    # temperature 0 + MTP heads: generate() itself defers to the speculative
    # path (observable via decode_heads, which the plain path never calls) and
    # still emits the exact greedy stream -- chat/API/eval get the speedup
    # without calling generate_speculative themselves.
    torch.manual_seed(0)
    tok = _tokenizer(tmp_path)
    model = _model(tok.n_vocab, n_mtp=2)
    prompt = tok.encode_ordinary("the quick brown fox")
    stop = {tok.encode_single_token(IM_END), tok.encode_single_token(EOS_TOKEN)}
    ref = _greedy_reference(model, prompt, budget=25, stop_ids=stop)

    calls = {"decode_heads": 0}
    orig = model.decode_heads

    def spy(*args, **kwargs):
        calls["decode_heads"] += 1
        return orig(*args, **kwargs)

    model.decode_heads = spy
    cfg = SamplingConfig(temperature=0.0, max_new_tokens=25)
    out = list(generate(model, tok, prompt, cfg, max_seq_len=128))
    assert out == ref
    assert calls["decode_heads"] > 0


def test_generate_sampling_stays_on_plain_path(tmp_path):
    # temperature > 0: speculative decoding would change the sampled
    # distribution (it is greedy-only), so generate() must not route there.
    torch.manual_seed(0)
    tok = _tokenizer(tmp_path)
    model = _model(tok.n_vocab, n_mtp=2)
    model.decode_heads = lambda *a, **kw: pytest.fail(
        "speculative path used for sampled decoding"
    )
    cfg = SamplingConfig(temperature=1.0, top_k=None, max_new_tokens=5)
    prompt = tok.encode_ordinary("the quick")
    out = list(generate(model, tok, prompt, cfg, max_seq_len=128))
    assert len(out) <= 5


def test_speculative_falls_back_without_mtp_heads(tmp_path):
    # n_mtp == 0: defer to plain greedy generate() (same stream).
    torch.manual_seed(0)
    tok = _tokenizer(tmp_path)
    model = _model(tok.n_vocab, n_mtp=0)
    prompt = tok.encode_ordinary("the quick")
    cfg = SamplingConfig(temperature=0.0, max_new_tokens=15)  # greedy generate()
    spec = list(generate_speculative(model, tok, prompt, cfg, max_seq_len=128))
    plain = list(generate(model, tok, prompt, cfg, max_seq_len=128))
    assert spec == plain
