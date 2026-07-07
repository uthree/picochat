import torch.nn as nn
import pytest
import torch

from picochat.model.gpt import (
    MODEL_PRESETS,
    TransformerLM,
    build_lm,
    estimate_num_params,
    estimate_preset_params,
    load_state_dict_expand,
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
    assert len(logits) == 1
    assert logits[0].shape == (2, 16, 50)


def test_build_lm_overrides_preset():
    lm = build_lm("pico", vocab_size=50, n_layers=2)
    assert lm.transformer.n_layers == 2  # overridden from preset's 8


def test_build_lm_vocab_override():
    lm = build_lm("pico", vocab_size=123)
    assert lm.embed.num_embeddings == 123
    assert all(head.out_features == 123 for head in lm.lmheads)


def test_build_lm_n_lmheads_override():
    lm = build_lm("pico", vocab_size=50, n_lmheads=2)
    assert len(lm.lmheads) == 2


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
        dict(vocab_size=50, d_model=48, n_heads=6, n_layers=2, n_lmheads=3),  # MTP
        dict(vocab_size=50, d_model=48, n_heads=6, n_layers=2, d_ffn=128, n_experts=4),
        dict(vocab_size=40, d_model=32, n_heads=4, n_layers=2, tie_embeddings=True),
        dict(
            vocab_size=50, d_model=48, n_heads=6, n_layers=2, n_lmheads=3,
            tie_embeddings=True,
        ),  # tied + MTP: only lmheads[0] is tied
    ],
)
def test_estimate_num_params_matches_actual(cfg):
    # the estimate mirrors the real module shapes exactly for these configs
    lm = TransformerLM(**cfg)
    assert estimate_num_params(**cfg) == _actual_params(lm)


def test_estimate_num_params_tie_embeddings_active_only_adds_nothing_extra():
    # active params only touch lmheads[0]; when tied it's already counted in
    # the embedding, so tied active count == untied active count minus one head.
    cfg = dict(vocab_size=50, d_model=48, n_heads=6, n_layers=2, n_lmheads=3)
    untied_active = estimate_num_params(**cfg, active_only=True)
    tied_active = estimate_num_params(**cfg, tie_embeddings=True, active_only=True)
    assert untied_active - tied_active == 50 * 48


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


def test_estimate_active_params_drops_inactive_experts_and_heads():
    cfg = dict(
        vocab_size=100,
        d_model=64,
        n_heads=8,
        n_layers=2,
        n_experts=8,
        d_expert=16,
        n_active=2,
        n_lmheads=3,
    )
    total = estimate_num_params(**cfg)
    active = estimate_num_params(**cfg, active_only=True)
    # active drops (n_experts - n_active) experts per layer and the extra MTP heads
    expert_drop = (8 - 2) * 3 * 16 * 64 * 2
    head_drop = (3 - 1) * 100 * 64
    assert active == total - expert_drop - head_drop
    assert active < total


def test_estimate_preset_active_params_smaller_than_total():
    for size in MODEL_PRESETS:
        total = estimate_preset_params(size)
        active = estimate_preset_params(size, active_only=True)
        if "n_experts" in MODEL_PRESETS[size]:
            # every preset is a MoE model, so active is a strict subset
            assert 0 < active < total


# ---------------------------------------------------------------------------
# load_state_dict_expand
# ---------------------------------------------------------------------------
def test_expand_load_matrix_copies_top_left_and_keeps_rest_random():
    small = nn.Linear(4, 3, bias=False)
    big = nn.Linear(6, 5, bias=False)
    torch.manual_seed(0)
    with torch.no_grad():
        small.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        big.weight.copy_(torch.full((5, 6), -1.0))  # sentinel "random init"

    stats = load_state_dict_expand(big, small.state_dict())

    assert stats == {"matched": 0, "expanded": 1, "skipped": 0}
    # top-left block came from the checkpoint
    assert torch.equal(big.weight[:3, :4], small.weight)
    # everything outside the top-left block is untouched (still the sentinel)
    assert torch.all(big.weight[3:, :] == -1.0)
    assert torch.all(big.weight[:, 4:] == -1.0)


def test_expand_load_missing_key_left_at_random_init():
    src = nn.Sequential(nn.Linear(2, 2, bias=False))
    dst = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        dst[1].weight.fill_(-1.0)

    stats = load_state_dict_expand(dst, src.state_dict())

    assert stats["skipped"] == 1  # "1.weight" has no counterpart in src
    assert torch.all(dst[1].weight == -1.0)  # untouched


def test_expand_load_same_shape_matches_exactly():
    lm_small = build_lm("pico", vocab_size=32, max_seq_len=64, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1)
    lm_other = build_lm("pico", vocab_size=32, max_seq_len=64, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1)

    stats = load_state_dict_expand(lm_other, lm_small.state_dict())

    assert stats["skipped"] == 0
    assert stats["expanded"] == 0
    assert stats["matched"] > 0
    for (k1, v1), (k2, v2) in zip(
        lm_small.state_dict().items(), lm_other.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(v1, v2)


def test_expand_load_grows_transformer_lm_d_model():
    torch.manual_seed(0)
    small = build_lm(
        "pico", vocab_size=32, max_seq_len=64, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1
    )
    big = build_lm(
        "pico", vocab_size=32, max_seq_len=64, d_model=32, n_layers=1, n_heads=4, n_kv_heads=1
    )

    stats = load_state_dict_expand(big, small.state_dict())

    assert stats["expanded"] > 0
    # the embedding's low-index corner was copied from the small checkpoint
    assert torch.equal(
        big.embed.weight[:, :16], small.embed.weight
    )
