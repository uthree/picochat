"""Growth / upcycling (picochat.model.grow): a small model's trained weights warm-start
a larger one. The headline property is function preservation -- at init the grown
model computes the (nearly) identical function -- so each test grows a small
model's state_dict, loads it into the larger model, and compares their outputs.
Everything runs on the CPU reference paths of the GDN / NSA mixers.
"""

import copy

import pytest
import torch

from picochat.model import TransformerLM
from picochat.model.grow import (
    grow_depth,
    grow_state_dict,
    grow_width,
    upcycle_to_moe,
)

# A tiny hybrid backbone: layers_per_block=2 -> alternating GDN (linear) and NSA
# (block-tail) layers, so both mixers are exercised. d_head = 8 is held constant
# across every width so HyperCloning applies.
BASE = dict(
    vocab_size=32,
    d_model=16,
    n_heads=2,  # d_head 8
    n_layers=4,
    n_kv_heads=1,
    nsa_kv_heads=1,
    layers_per_block=2,
    d_ffn=48,
)


def _build(cfg, seed=0, max_seq_len=64):
    torch.manual_seed(seed)
    return TransformerLM(max_seq_len=max_seq_len, **cfg).eval()


def _out(lm, x):
    with torch.no_grad():
        return lm(x)


def _rel_err(a, b):
    return (a - b).norm() / b.norm().clamp_min(1e-9)


@pytest.fixture
def x():
    torch.manual_seed(123)
    return torch.randint(0, BASE["vocab_size"], (2, 24))


# ---------------------------------------------------------------------------
# width (HyperCloning) -- exactly function-preserving
# ---------------------------------------------------------------------------
def _widen(cfg, r):
    """The r-widened config: d_head fixed, every width field scaled by r."""
    out = dict(cfg)
    out["d_model"] = cfg["d_model"] * r
    for f in ("n_heads", "n_kv_heads", "nsa_kv_heads", "d_ffn"):
        out[f] = cfg[f] * r
    return out


@pytest.mark.parametrize("r", [2, 3])
def test_grow_width_is_function_preserving(x, r):
    src_cfg, tgt_cfg = dict(BASE), _widen(BASE, r)
    src = _build(src_cfg, seed=1)
    grown = grow_width(src.state_dict(), *(_norm(src_cfg), _norm(tgt_cfg)))
    tgt = _build(tgt_cfg, seed=2)  # random init, about to be overwritten
    missing, unexpected = tgt.load_state_dict(grown, strict=False)
    assert not missing and not unexpected  # keys line up exactly
    assert _rel_err(_out(tgt, x), _out(src, x)) < 1e-4


def test_grow_width_symmetry_broken(x):
    # the r copies of each widened weight must NOT be identical (else they share
    # gradients forever); the row-sum-zero noise keeps the function preserved
    # while breaking the tie.
    src_cfg = dict(BASE)
    src = _build(src_cfg, seed=1)
    grown = grow_width(
        src.state_dict(), _norm(src_cfg), _norm(_widen(BASE, 2)), noise=0.02
    )
    w = grown["transformer.layers.0.ffn.proj_up.weight"]  # (2*d_ffn, 2*d_model)
    d_ffn = BASE["d_ffn"]
    top, bot = w[:d_ffn], w[d_ffn:]  # the two output copies
    assert not torch.allclose(top, bot)


# ---------------------------------------------------------------------------
# depth (block stacking) -- approximately function-preserving
# ---------------------------------------------------------------------------
def test_grow_depth_keys_match_and_near_identity(x):
    src_cfg = dict(BASE)
    tgt_cfg = dict(BASE, n_layers=6)  # +1 block (layers_per_block=2)
    src = _build(src_cfg, seed=1)
    grown = grow_depth(src.state_dict(), _norm(src_cfg), _norm(tgt_cfg))
    tgt = _build(tgt_cfg, seed=2)
    missing, unexpected = tgt.load_state_dict(grown, strict=False)
    assert not missing and not unexpected

    # appended layers write nothing into the residual at init
    z = grown["transformer.layers.4.ffn.proj_down.weight"]
    assert torch.count_nonzero(z) == 0
    assert torch.count_nonzero(grown["transformer.layers.5.attn.proj_o.weight"]) == 0

    # grown output is far closer to the source than a randomly-initialized model
    # of the target shape (the appended blocks perturb only the DepthAttention
    # renormalization).
    rand = _build(tgt_cfg, seed=7)
    src_o = _out(src, x)
    assert _rel_err(_out(tgt, x), src_o) < 0.5 * _rel_err(_out(rand, x), src_o)


# ---------------------------------------------------------------------------
# dense -> MoE (sparse upcycling) -- function-preserving when d_ffn is kept
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "moe",
    [
        dict(n_experts=4, n_active=2, d_expert=12, d_latent=8),  # LatentMoE
        dict(n_experts=4, n_active=2, d_expert=48),  # non-latent
        dict(n_experts=4, n_active=2, d_expert=48, share_experts=True),  # shared pool
    ],
)
def test_upcycle_to_moe_is_function_preserving(x, moe):
    src_cfg = dict(BASE)
    tgt_cfg = dict(BASE, **moe)  # d_ffn (48) preserved as the shared FFN
    src = _build(src_cfg, seed=1)
    grown = upcycle_to_moe(src.state_dict(), _norm(src_cfg), _norm(tgt_cfg))
    tgt = _build(tgt_cfg, seed=2)
    missing, unexpected = tgt.load_state_dict(grown, strict=False)
    assert not missing and not unexpected
    # the routed branch is zero at init -> the MoE model == the dense one
    assert _rel_err(_out(tgt, x), _out(src, x)) < 1e-5


def test_upcycle_smaller_ffn_preserves_backbone_not_exact(x):
    # a smaller shared FFN can't be function-preserving, but the attention
    # backbone / embeddings must still be carried over verbatim.
    src_cfg = dict(BASE, d_ffn=48)
    tgt_cfg = dict(BASE, d_ffn=16, n_experts=4, n_active=2, d_expert=12, d_latent=8)
    src = _build(src_cfg, seed=1)
    grown = upcycle_to_moe(src.state_dict(), _norm(src_cfg), _norm(tgt_cfg))
    assert torch.equal(grown["embed.weight"], src.state_dict()["embed.weight"])
    assert torch.equal(
        grown["transformer.layers.0.attn.proj_q.weight"],
        src.state_dict()["transformer.layers.0.attn.proj_q.weight"],
    )
    tgt = _build(tgt_cfg, seed=2)
    assert tgt.load_state_dict(grown, strict=True)  # shapes all line up


# ---------------------------------------------------------------------------
# composition (grow_state_dict) + MTP head adjustment
# ---------------------------------------------------------------------------
def test_grow_state_dict_width_then_depth(x):
    src_cfg = dict(BASE)
    tgt_cfg = dict(_widen(BASE, 2), n_layers=6)  # wider AND deeper
    src = _build(src_cfg, seed=1)
    grown = grow_state_dict(src.state_dict(), src_cfg, tgt_cfg)
    tgt = _build(tgt_cfg, seed=2)
    tgt.load_state_dict(grown, strict=True)  # raises if any key/shape is off
    # width is exact; the extra block only perturbs the residual mix a little
    assert _rel_err(_out(tgt, x), _out(src, x)) < 0.2


def test_grow_state_dict_adds_mtp_heads(x):
    src_cfg = dict(BASE)
    tgt_cfg = dict(BASE, n_mtp=2)
    src = _build(src_cfg, seed=1)
    grown = grow_state_dict(src.state_dict(), src_cfg, tgt_cfg)
    tgt = _build(tgt_cfg, seed=2)
    tgt.load_state_dict(grown, strict=True)
    assert len(tgt.mtp_heads) == 2
    # MTP heads are the identity at init, so the primary-head logits are unchanged
    assert _rel_err(_out(tgt, x), _out(src, x)) < 1e-4


def test_grow_full_pipeline_dense_to_moe(x):
    # width -> depth -> MTP -> upcycle, all at once, into a real target model
    src_cfg = dict(BASE)
    tgt_cfg = dict(
        _widen(BASE, 2),
        n_layers=6,
        n_mtp=1,
        n_experts=4,
        n_active=2,
        d_expert=24,
        d_latent=16,
        d_ffn=BASE["d_ffn"] * 2,  # width-scaled; kept as the shared FFN
    )
    src = _build(src_cfg, seed=1)
    grown = grow_state_dict(src.state_dict(), src_cfg, tgt_cfg)
    tgt = _build(tgt_cfg, seed=2)
    tgt.load_state_dict(grown, strict=True)
    assert torch.isfinite(_out(tgt, x)).all()


def test_grow_real_preset_widths_200m_to_1b():
    # Exercise the actual ladder's width fields (d_model 1024->2048, n_heads
    # 16->32, n_kv 4->8, nsa_kv 1->2, d_ffn 2048->4096): grow the real 200m
    # preset into the real 1b preset and load strict. Layer counts are cut (real
    # width, few layers) so the 2048-wide build stays light.
    from picochat.model import TransformerLM
    from picochat.model.presets import resolve_config

    src_cfg = resolve_config("200m", vocab_size=128, n_layers=4)
    tgt_cfg = resolve_config("1b", vocab_size=128, n_layers=8)
    src = TransformerLM(max_seq_len=64, **src_cfg).eval()
    grown = grow_state_dict(src.state_dict(), src_cfg, tgt_cfg)
    tgt = TransformerLM(max_seq_len=64, **tgt_cfg).eval()
    tgt.load_state_dict(grown, strict=True)  # keys + shapes match build_lm("1b")
    xs = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        assert _rel_err(tgt(xs), src(xs)) < 1e-3  # width is exact, depth near-exact


def test_grow_moe_to_moe_rejected():
    src_cfg = dict(BASE, n_experts=4, n_active=2, d_expert=12)
    tgt_cfg = dict(BASE, n_experts=8, n_active=2, d_expert=12)
    with pytest.raises(ValueError):
        grow_state_dict({}, src_cfg, tgt_cfg)


def test_grow_width_requires_constant_d_head():
    # widening the head size (not the count) is not function-preserving -> refuse
    src_cfg = dict(BASE)  # d_head 8
    bad = dict(BASE, d_model=32, n_heads=2)  # d_head 16 (doubled)
    with pytest.raises(AssertionError):
        grow_width({}, _norm(src_cfg), _norm(bad))


# small local mirror of grow._norm_cfg so tests read declaratively
def _norm(cfg):
    from picochat.model.grow import _norm_cfg

    return _norm_cfg(copy.deepcopy(cfg))
