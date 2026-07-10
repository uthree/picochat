import pytest
import torch

from picochat.model.gpt import Transformer, TransformerLM, estimate_num_params


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
    )
    tokens = torch.randint(0, vocab_size, (2, 5))
    logits = lm(tokens)
    assert logits.shape == (2, 5, vocab_size)


def test_transformer_lm_incremental_matches_full():
    vocab_size = 40
    lm = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2).eval()
    tokens = torch.randint(0, vocab_size, (1, 5))
    full = lm(tokens)

    _, cache, pos = lm.decode(tokens[:, :4])
    step, _, _ = lm.decode(tokens[:, 4:5], cache, pos)
    assert torch.allclose(full[:, 4:5], step, atol=1e-4)


# ---------------------------------------------------------------------------
# Weight init: every weight ~ normal(0, init_std), uniformly (no depth/residual
# scaling) so a checkpoint expanded via load_state_dict_expand into a larger
# config sees statistically identical random init in the untouched region
# regardless of which layer it lands in.
# ---------------------------------------------------------------------------
def test_init_uses_configured_std_uniformly():
    torch.manual_seed(0)
    init_std = 0.01
    lm = TransformerLM(
        vocab_size=2000, d_model=64, n_heads=8, n_layers=6, init_std=init_std
    )
    # A layer-0 residual projection and a layer-5 one must share the same std
    # -- there's no more 1/sqrt(2*n_layers) depth scaling.
    proj_o_0 = lm.transformer.layers[0].attn.proj_o.weight
    proj_o_5 = lm.transformer.layers[5].attn.proj_o.weight
    proj_down_0 = lm.transformer.layers[0].ffn.proj_down.weight
    for w in (proj_o_0, proj_o_5, proj_down_0, lm.embed.weight):
        assert w.std().item() == pytest.approx(init_std, rel=0.15)


def test_init_default_std_is_smaller_than_gpt2_style():
    # 0.01 (well under GPT-2's canonical 0.02) since checkpoints get expanded
    # into larger configs via load_state_dict_expand, and a smaller std keeps
    # the freshly-random region closer in scale to the loaded region.
    lm = TransformerLM(vocab_size=100, d_model=32, n_heads=4, n_layers=2)
    assert lm.init_std == pytest.approx(0.01)
    assert lm.init_std < 0.02


def test_init_biases_are_zero():
    lm = TransformerLM(vocab_size=100, d_model=32, n_heads=4, n_layers=2)
    for m in lm.modules():
        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            assert torch.all(m.bias == 0)


# ---------------------------------------------------------------------------
# estimate_num_params
# ---------------------------------------------------------------------------
def _actual_params(lm) -> int:
    return sum(p.numel() for p in lm.parameters())


@pytest.mark.parametrize(
    "cfg",
    [
        dict(vocab_size=50, d_model=32, n_heads=4, n_layers=3),  # dense
        dict(vocab_size=50, d_model=64, n_heads=8, n_layers=3,
             n_experts=4, d_expert=16),  # MoE
        dict(vocab_size=50, d_model=48, n_heads=6, n_layers=2,
             tie_embeddings=True),  # tied embeddings
    ],
)
def test_estimate_num_params_matches_actual(cfg):
    lm = TransformerLM(**cfg)
    assert estimate_num_params(**cfg) == _actual_params(lm)
