import pytest
import torch

from picochat.model.gpt import (
    MODEL_PRESETS,
    TransformerLM,
    build_lm,
    estimate_num_params,
    estimate_preset_params,
)


# ---------------------------------------------------------------------------
# scale-ladder presets
# ---------------------------------------------------------------------------
def test_build_lm_unknown_size_raises():
    with pytest.raises(ValueError):
        build_lm("gigantic", vocab_size=32)


@pytest.mark.parametrize("size", list(MODEL_PRESETS))
def test_preset_dims_are_consistent(size):
    cfg = MODEL_PRESETS[size]
    assert cfg["d_model"] % cfg["n_heads"] == 0  # heads tile d_model
    assert cfg["n_heads"] % cfg["n_kv_heads"] == 0  # GQA grouping
    assert (cfg["d_model"] // cfg["n_heads"]) % 2 == 0  # d_head even (RoPE)


def test_build_lm_pico_forward():
    lm = build_lm("pico", vocab_size=50, max_seq_len=64)
    logits = lm(torch.randint(0, 50, (2, 16)))
    assert logits.shape == (2, 16, 50)


def test_build_lm_overrides_preset():
    lm = build_lm("pico", vocab_size=50, n_layers=2)
    assert lm.transformer.n_layers == 2  # overridden from preset's 8


def test_build_lm_vocab_override():
    lm = build_lm("pico", vocab_size=123)
    assert lm.embed.num_embeddings == 123
    assert lm.lmhead.out_features == 123


# ---------------------------------------------------------------------------
# estimate_num_params
# ---------------------------------------------------------------------------
def _actual_params(lm) -> int:
    return sum(p.numel() for p in lm.parameters())


@pytest.mark.parametrize(
    "cfg",
    [
        dict(vocab_size=40, d_model=32, n_heads=4, n_layers=2),  # dense MHA
        dict(vocab_size=100, d_model=64, n_heads=8, n_kv_heads=2, n_layers=3),  # GQA
        dict(
            vocab_size=100, d_model=64, n_heads=8, n_layers=2, n_experts=4, d_expert=16
        ),
        dict(vocab_size=50, d_model=48, n_heads=6, n_layers=2, d_ffn=128, n_experts=4),
        dict(vocab_size=40, d_model=32, n_heads=4, n_layers=2, d_ffn=96),  # d_ffn set
    ],
)
def test_estimate_num_params_matches_actual(cfg):
    # the estimate mirrors the real module shapes exactly for these configs
    lm = TransformerLM(**cfg)
    assert estimate_num_params(**cfg) == _actual_params(lm)


def test_estimate_num_params_counts_untied_head():
    # the lm head is a separate (untied) projection, so it and the embedding
    # each contribute vocab * d_model.
    cfg = dict(vocab_size=50, d_model=48, n_heads=6, n_layers=2)
    lm = TransformerLM(**cfg)
    assert lm.lmhead.weight is not lm.embed.weight
    assert estimate_num_params(**cfg) == _actual_params(lm)


def test_estimate_num_params_ignores_extra_kwargs():
    # a preset / saved model_config carries non-shape keys (max_seq_len,
    # window_size, ...) that must be accepted and ignored so it can be splatted
    base = dict(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    ref = estimate_num_params(**base)
    extra = dict(
        **base, max_seq_len=4096, window_size=128, global_attn_ratio=4, rope_base=10000
    )
    assert estimate_num_params(**extra) == ref


def test_estimate_preset_params_matches_build_lm():
    # same preset/override resolution as build_lm -> same model -> same count
    lm = build_lm("pico", vocab_size=1000, n_layers=2)
    assert estimate_preset_params(
        "pico", vocab_size=1000, n_layers=2
    ) == _actual_params(lm)


def test_estimate_preset_params_all_presets_positive_and_monotone():
    counts = [estimate_preset_params(s) for s in MODEL_PRESETS]
    assert all(c > 0 for c in counts)
    # the ladder is listed smallest -> largest
    assert counts == sorted(counts)


def test_estimate_preset_params_unknown_raises():
    with pytest.raises(ValueError):
        estimate_preset_params("gigantic")


def test_estimate_active_params_equals_total_for_dense():
    # no experts, single head -> nothing to sparsify
    cfg = dict(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    assert estimate_num_params(**cfg, active_only=True) == estimate_num_params(**cfg)


def test_estimate_active_params_drops_inactive_experts():
    cfg = dict(
        vocab_size=100,
        d_model=64,
        n_heads=8,
        n_layers=2,
        n_experts=8,
        d_expert=16,
        n_active=2,
    )
    total = estimate_num_params(**cfg)
    active = estimate_num_params(**cfg, active_only=True)
    # active drops (n_experts - n_active) experts per layer
    expert_drop = (8 - 2) * 3 * 16 * 64 * 2
    assert active == total - expert_drop
    assert active < total


def test_estimate_preset_active_params_smaller_than_total():
    for size in MODEL_PRESETS:
        total = estimate_preset_params(size)
        active = estimate_preset_params(size, active_only=True)
        if "n_experts" in MODEL_PRESETS[size]:
            # every preset is a MoE model, so active is a strict subset
            assert 0 < active < total
