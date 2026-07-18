import pytest
import torch

from picochat.gpt import (
    MODEL_PRESETS,
    TransformerLM,
    build_lm,
    estimate_num_params,
    estimate_preset_params,
)


# ---------------------------------------------------------------------------
# scale-ladder presets (loaded from configs/presets.yml)
# ---------------------------------------------------------------------------
# The ladder is two axes: a total-param rung (200m/1b/8b/35b/120b) crossed with
# an architecture variant (dense / -moe / -moe-shared). Dense stops at 8b; MoE
# starts at 1b (see configs/presets.yml).
EXPECTED_PRESETS = [
    "200m",
    "1b",
    "1b-moe",
    "1b-moe-shared",
    "8b",
    "8b-moe",
    "8b-moe-shared",
    "35b-moe",
    "35b-moe-shared",
    "120b-moe",
    "120b-moe-shared",
]
RUNGS = ["200m", "1b", "8b", "35b", "120b"]


def _rung(size: str) -> str:
    # the leading token before the first '-' is the total-param rung
    return size.split("-", 1)[0]


def test_presets_load_from_yaml_with_required_keys():
    # the ladder lives in configs/presets.yml; a malformed edit there (missing
    # key, non-integer value) must fail loudly here rather than at train time
    assert list(MODEL_PRESETS) == EXPECTED_PRESETS
    required = {"d_model", "n_layers", "n_heads", "n_kv_heads", "vocab_size"}
    for size, cfg in MODEL_PRESETS.items():
        assert required <= cfg.keys(), f"{size} is missing {required - cfg.keys()}"
        # every value is an int (share_experts is a bool, an int subclass)
        assert all(isinstance(v, int) for v in cfg.values()), size


def test_preset_names_match_dense_and_moe_axes():
    # dense only up to 8b; -moe / -moe-shared for every rung from 1b up
    for size, cfg in MODEL_PRESETS.items():
        if size.endswith("-moe-shared"):
            assert cfg.get("share_experts") is True and "n_experts" in cfg
        elif size.endswith("-moe"):
            assert "n_experts" in cfg and not cfg.get("share_experts", False)
        else:  # dense
            assert "n_experts" not in cfg
    assert "200m" in MODEL_PRESETS and "200m-moe" not in MODEL_PRESETS  # tiny is dense
    assert "35b" not in MODEL_PRESETS  # no dense past 8b
    assert "120b" not in MODEL_PRESETS


def test_nonshared_moe_is_latent_and_fine_grained():
    # every -moe (non-shared) preset is LatentMoE (d_latent set) and fine-grained
    # (many small experts, high top-k) -- see configs/presets.yml.
    for size, cfg in MODEL_PRESETS.items():
        if size.endswith("-moe"):  # non-shared MoE
            assert cfg.get("d_latent"), f"{size} should set d_latent (LatentMoE)"
            assert cfg["n_active"] >= 8, size  # fine-grained: high top-k


def test_shared_moe_is_coarse_grained_and_not_latent():
    # every -moe-shared is coarse-grained (few-but-large experts, low top-k) with
    # no latent compression, and larger experts + lower top-k than its non-shared
    # sibling at the same rung.
    for size, cfg in MODEL_PRESETS.items():
        if not size.endswith("-moe-shared"):
            continue
        assert "d_latent" not in cfg, f"{size} shared pool is not LatentMoE"
        assert cfg["n_active"] <= 2, size  # coarse: low top-k
        sibling = MODEL_PRESETS[size[: -len("-shared")]]  # the -moe at this rung
        assert cfg["d_expert"] > sibling["d_expert"]  # coarser (bigger) experts
        assert cfg["n_active"] < sibling["n_active"]


def test_moe_variants_match_rung_total():
    # -moe and -moe-shared at the same rung are sized to the same total (within a
    # tolerance); active differs (that is the point of comparing them).
    rungs = {}
    for size in MODEL_PRESETS:
        if size.endswith("-moe") or size.endswith("-moe-shared"):
            rungs.setdefault(_rung(size), []).append(size)
    for rung, names in rungs.items():
        totals = [estimate_preset_params(n) for n in names]
        assert max(totals) / min(totals) < 1.15, f"{rung}: {names} totals diverge"


def test_build_lm_unknown_size_raises():
    with pytest.raises(ValueError):
        build_lm("gigantic", vocab_size=32)


@pytest.mark.parametrize("size", list(MODEL_PRESETS))
def test_preset_dims_are_consistent(size):
    cfg = MODEL_PRESETS[size]
    assert cfg["d_model"] % cfg["n_heads"] == 0  # heads tile d_model
    assert cfg["n_heads"] % cfg["n_kv_heads"] == 0  # GQA grouping
    assert (cfg["d_model"] // cfg["n_heads"]) % 2 == 0  # d_head even (RoPE)


def test_build_lm_smallest_forward():
    lm = build_lm("200m", vocab_size=50, max_seq_len=64)
    logits = lm(torch.randint(0, 50, (2, 16)))
    assert logits.shape == (2, 16, 50)


def test_build_lm_overrides_preset():
    lm = build_lm("200m", vocab_size=50, n_layers=2)
    assert lm.transformer.n_layers == 2  # overridden from preset's 12


def test_build_lm_vocab_override():
    lm = build_lm("200m", vocab_size=123)
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
        dict(  # LatentMoE: experts in a d_latent-dim space
            vocab_size=100,
            d_model=64,
            n_heads=8,
            n_layers=2,
            n_experts=8,
            d_expert=24,
            d_latent=16,
        ),
        dict(  # shared routed-expert bank across layers
            vocab_size=100,
            d_model=64,
            n_heads=8,
            n_layers=3,
            n_experts=4,
            d_expert=16,
            share_experts=True,
        ),
        dict(  # shared bank + latent experts
            vocab_size=100,
            d_model=64,
            n_heads=8,
            n_layers=2,
            n_experts=8,
            d_expert=24,
            d_latent=16,
            share_experts=True,
        ),
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
        **base, max_seq_len=4096, window_size=128, layers_per_block=4, rope_base=10000
    )
    assert estimate_num_params(**extra) == ref


def test_estimate_preset_params_matches_build_lm():
    # same preset/override resolution as build_lm -> same model -> same count
    lm = build_lm("200m", vocab_size=1000, n_layers=2)
    assert estimate_preset_params(
        "200m", vocab_size=1000, n_layers=2
    ) == _actual_params(lm)


def test_estimate_preset_params_positive_and_rungs_are_ordered():
    counts = {s: estimate_preset_params(s) for s in MODEL_PRESETS}
    assert all(c > 0 for c in counts.values())
    # totals aren't globally monotone (dense/-moe/-moe-shared cluster within a
    # rung), but the rungs themselves climb: every preset on a higher rung is
    # bigger than every preset on a lower one.
    for lo, hi in zip(RUNGS, RUNGS[1:]):
        lo_max = max(c for s, c in counts.items() if _rung(s) == lo)
        hi_min = min(c for s, c in counts.items() if _rung(s) == hi)
        assert lo_max < hi_min, f"{lo} rung overlaps {hi} rung"


def test_estimate_preset_params_unknown_raises():
    with pytest.raises(ValueError):
        estimate_preset_params("gigantic")


def test_estimate_active_params_equals_total_for_dense():
    # no experts, single head -> nothing to sparsify
    cfg = dict(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    assert estimate_num_params(**cfg, active_only=True) == estimate_num_params(**cfg)


def test_estimate_active_params_shared_bank_saturates():
    # with per-layer routing into one shared pool, a token can touch up to
    # n_layers * n_active distinct experts, capped by the pool size. Here
    # 4 layers * 2 active = 8 > 4 experts, so the whole bank counts as active
    # and (everything else being dense) active equals total.
    cfg = dict(
        vocab_size=100,
        d_model=64,
        n_heads=8,
        n_layers=4,
        n_experts=4,
        d_expert=16,
        n_active=2,
        share_experts=True,
    )
    assert estimate_num_params(**cfg, active_only=True) == estimate_num_params(**cfg)
    # a large pool doesn't saturate: only n_layers * n_active experts count
    big = dict(cfg, n_experts=32)
    total, active = estimate_num_params(**big), estimate_num_params(**big, active_only=True)
    assert total - active == (32 - 4 * 2) * 3 * 16 * 64


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
