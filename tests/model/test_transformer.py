import pytest
import torch

from picochat.model.gpt import GPT, Transformer, TransformerLM, estimate_num_params


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
def test_transformer_output_shape():
    model = Transformer(d_model=32, n_heads=4, n_layers=3)
    x = torch.randn(2, 7, 32)
    out = model(x)
    assert out.shape == x.shape


def test_transformer_cache_per_layer():
    n_layers = 3
    model = Transformer(d_model=32, n_heads=4, n_layers=n_layers)
    out, cache, pos = model.decode(torch.randn(2, 7, 32))
    assert len(cache) == n_layers
    assert all(c is not None for c in cache)


def test_transformer_decode_pos_advances():
    model = Transformer(d_model=32, n_heads=4, n_layers=2)
    _, cache, pos = model.decode(torch.randn(2, 5, 32))
    assert pos == 5
    _, cache, pos = model.decode(torch.randn(2, 3, 32), cache, pos)
    assert pos == 8


def test_transformer_incremental_matches_full():
    torch.manual_seed(0)
    model = Transformer(d_model=32, n_heads=4, n_layers=2).eval()
    x = torch.randn(1, 5, 32)
    full = model(x)

    out, cache, pos = model.decode(x[:, :4])
    step, _, _ = model.decode(x[:, 4:5], cache, pos)
    assert torch.allclose(full[:, 4:5], step, atol=1e-4)


def test_transformer_grouped_query():
    model = Transformer(d_model=32, n_heads=8, n_layers=2, n_kv_heads=2)
    out, cache, pos = model.decode(torch.randn(2, 4, 32))
    assert out.shape == (2, 4, 32)
    assert cache[0].shape[2] == 2  # n_kv_heads heads cached


def test_transformer_backward():
    model = Transformer(d_model=32, n_heads=4, n_layers=2)
    x = torch.randn(2, 5, 32, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# TransformerLM
# ---------------------------------------------------------------------------
def test_transformer_lm_logits_shape():
    vocab_size = 40
    lm = TransformerLM(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_coda_layers=1,
        n_prelude_layers=1,
        n_recursions=2,
    )
    tokens = torch.randint(0, vocab_size, (2, 5))
    logits = lm(tokens)
    # forward returns one logits tensor per lm head (a single head by default)
    assert len(logits) == 1
    assert logits[0].shape == (2, 5, vocab_size)


def test_transformer_lm_incremental_matches_full():
    vocab_size = 40
    lm = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2).eval()
    tokens = torch.randint(0, vocab_size, (1, 5))
    full = lm(tokens)[0]

    _, cache, pos = lm.decode(tokens[:, :4])
    step, _, _ = lm.decode(tokens[:, 4:5], cache, pos)
    assert torch.allclose(full[:, 4:5], step, atol=1e-4)


# ---------------------------------------------------------------------------
# Looped LM (prelude - recursed middle - coda)
# ---------------------------------------------------------------------------
def _looped(**over):
    cfg = dict(
        d_model=32, n_heads=4, n_layers=3,
        n_prelude_layers=2, n_coda_layers=1, n_recursions=2, global_attn_ratio=1,
    )
    cfg.update(over)
    return Transformer(**cfg)


def test_looped_layer_counts():
    m = _looped()
    assert len(m.prelude_layers) == 2
    assert len(m.coda_layers) == 1
    assert len(m.layers) == 3  # the recursed middle block is stored once
    assert m.n_recursions == 2


def test_looped_forward_shape_and_backward():
    m = _looped()
    x = torch.randn(2, 6, 32, requires_grad=True)
    out = m(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_recursion_reuses_middle_weights_no_extra_params():
    # the middle block is looped (shared weights), so param count is independent
    # of n_recursions -- that is the whole point of the recursion.
    base = dict(d_model=32, n_heads=4, n_layers=3, n_prelude_layers=1, n_coda_layers=1)
    n1 = sum(p.numel() for p in _looped(**base, n_recursions=1).parameters())
    n5 = sum(p.numel() for p in _looped(**base, n_recursions=5).parameters())
    assert n1 == n5


def test_recursion_actually_loops_changes_output():
    # more recursions must change the computation (else the loop is a no-op)
    torch.manual_seed(0)
    m1 = _looped(n_recursions=1).eval()
    m3 = _looped(n_recursions=3).eval()
    # give them identical weights so only the recursion count differs
    m3.load_state_dict(m1.state_dict())
    x = torch.randn(1, 5, 32)
    assert not torch.allclose(m1(x), m3(x), atol=1e-4)


def test_prelude_and_coda_have_no_moe_middle_does():
    m = _looped(n_experts=4, d_expert=16)
    assert all(not hasattr(l, "moe") for l in m.prelude_layers)
    assert all(not hasattr(l, "moe") for l in m.coda_layers)
    assert all(hasattr(l, "moe") for l in m.layers)


def test_looped_decode_cache_length_and_incremental_matches_full():
    # regression: the middle cache slot is i*n_layers+j (not i*n_recursions+j);
    # with n_layers != n_recursions a wrong stride collides/crashes.
    torch.manual_seed(0)
    m = _looped(n_layers=3, n_recursions=2, n_prelude_layers=1, n_coda_layers=1).eval()
    x = torch.randn(1, 5, 32)
    full = m(x)

    out, cache, pos = m.decode(x[:, :4])
    # one slot per prelude, per (recursion, middle layer), and per coda layer
    assert len(cache) == 1 + 3 * 2 + 1
    assert all(c is not None for c in cache)
    step, _, _ = m.decode(x[:, 4:5], cache, pos)
    assert torch.allclose(full[:, 4:5], step, atol=1e-4)


def test_looped_transformer_lm_incremental_matches_full():
    vocab = 40
    lm = TransformerLM(
        vocab_size=vocab, d_model=32, n_heads=4, n_layers=3,
        n_prelude_layers=1, n_coda_layers=1, n_recursions=2, global_attn_ratio=1,
    ).eval()
    tokens = torch.randint(0, vocab, (1, 6))
    full = lm(tokens)[0]
    _, cache, pos = lm.decode(tokens[:, :5])
    step, _, _ = lm.decode(tokens[:, 5:6], cache, pos)
    assert torch.allclose(full[:, 5:6], step, atol=1e-4)


def test_looped_gpt_generate_runs():
    lm = TransformerLM(
        vocab_size=40, d_model=32, n_heads=4, n_layers=3, max_seq_len=64,
        n_prelude_layers=1, n_coda_layers=1, n_recursions=3, global_attn_ratio=1,
    )
    gpt = GPT(lm, pad_idx=0, compile=False).eval()
    generated = gpt._generate(torch.randint(1, 40, (1, 8)), max_new_tokens=6)
    assert generated.shape == (1, 6)


# ---------------------------------------------------------------------------
# estimate_num_params with the looped structure
# ---------------------------------------------------------------------------
def _actual_params(lm) -> int:
    return sum(p.numel() for p in lm.parameters())


@pytest.mark.parametrize(
    "cfg",
    [
        dict(vocab_size=50, d_model=32, n_heads=4, n_layers=3,
             n_prelude_layers=2, n_coda_layers=2, n_recursions=4),  # dense looped
        dict(vocab_size=50, d_model=64, n_heads=8, n_layers=3, n_experts=4, d_expert=16,
             n_prelude_layers=1, n_coda_layers=1, n_recursions=3),  # MoE looped
        dict(vocab_size=50, d_model=48, n_heads=6, n_layers=2, n_lmheads=3,
             n_prelude_layers=1, n_coda_layers=2, n_recursions=2),  # MTP + looped
    ],
)
def test_estimate_num_params_matches_actual_looped(cfg):
    lm = TransformerLM(**cfg)
    assert estimate_num_params(**cfg) == _actual_params(lm)


def test_estimate_num_params_ignores_n_recursions():
    cfg = dict(vocab_size=50, d_model=32, n_heads=4, n_layers=3,
               n_prelude_layers=2, n_coda_layers=2)
    assert estimate_num_params(**cfg, n_recursions=1) == estimate_num_params(
        **cfg, n_recursions=9
    )
