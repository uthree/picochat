import copy

import pytest
import torch

from picochat.gpt import MixtureOfExperts, Transformer


def _moe_transformer(grad_checkpoint: bool) -> Transformer:
    return Transformer(
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_experts=4,
        d_expert=16,
        layers_per_block=1,  # all full attention -> CPU-friendly, no windows
        grad_checkpoint=grad_checkpoint,
    )


# ---------------------------------------------------------------------------
# MoE load-balancing bias under gradient checkpointing
# ---------------------------------------------------------------------------
def test_moe_bias_updates_once_per_training_forward():
    m = _moe_transformer(grad_checkpoint=False).train()
    moe = m.layers[0].moe
    x = torch.randn(2, 8, 32)
    before = moe.expert_bias.clone()
    m(x)
    # exactly one step of +/- bias_update_rate per expert
    delta = moe.expert_bias - before
    assert delta.abs().max() == pytest.approx(moe.bias_update_rate)
    assert (delta != 0).any()


def test_moe_bias_identical_with_and_without_checkpoint():
    # regression: an in-place update inside the checkpointed forward would run
    # twice (forward + backward recompute) and double the delta. Staging it and
    # applying once outside the checkpoint keeps both paths identical.
    torch.manual_seed(0)
    base = _moe_transformer(grad_checkpoint=False)
    x = torch.randn(2, 8, 32, requires_grad=True)

    def run(model):
        # reseed so dropout masks match across the two runs: a deeper layer's
        # MoE routing depends on the (dropout-affected) output of earlier layers,
        # so the comparison is only meaningful with the RNG aligned. Within a run,
        # checkpoint's own preserve_rng_state keeps forward/recompute consistent.
        torch.manual_seed(123)
        model = model.train()
        out = model(x)
        out.sum().backward()
        return [layer.moe.expert_bias.clone() for layer in model.layers]

    off = run(copy.deepcopy(base))
    on_model = copy.deepcopy(base)
    on_model.grad_checkpoint = True
    on = run(on_model)

    assert any((b != 0).any() for b in off)  # the update actually happened
    for a, b in zip(on, off):
        assert torch.allclose(a, b)  # checkpoint on == off (single update)
        # and each moved by exactly one rate step, not two
        assert a.abs().max() == pytest.approx(base.layers[0].moe.bias_update_rate)


def test_moe_bias_frozen_in_eval():
    m = _moe_transformer(grad_checkpoint=False).eval()
    moe = m.layers[0].moe
    before = moe.expert_bias.clone()
    m(torch.randn(2, 8, 32))
    assert torch.equal(moe.expert_bias, before)  # no drift outside training


def test_moe_pending_delta_not_in_state_dict():
    # the staging buffer is per-step scratch, not persistent state
    m = _moe_transformer(grad_checkpoint=False)
    keys = m.state_dict().keys()
    assert not any(k.endswith("_pending_bias_delta") for k in keys)
    assert any(k.endswith("expert_bias") for k in keys)  # the real buffer persists


# ---------------------------------------------------------------------------
# MoE at inference (decode must apply the experts, and never drop tokens)
# ---------------------------------------------------------------------------
def test_moe_decode_matches_full_forward():
    # regression: TransformerLayer.decode used to skip the MoE entirely, so a
    # sparse model generated from an FFN-only network. decode must now apply the
    # experts. n_experts=2/n_active=2 makes the bounded capacity >= n_tokens, so
    # the (dropping) full forward drops nothing here and matches decode exactly.
    torch.manual_seed(0)
    m = Transformer(
        d_model=32,
        n_heads=4,
        n_layers=3,
        n_experts=2,
        d_expert=16,
        n_active=2,
        layers_per_block=1,
    ).eval()
    x = torch.randn(1, 6, 32)
    full = m(x)

    out, cache, pos = m.decode(x[:, :4])  # prefill
    outs = [out]
    for t in range(4, 6):  # then step token by token
        o, cache, pos = m.decode(x[:, t : t + 1], cache, pos)
        outs.append(o)
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=1e-4)


def test_moe_no_drop_flag_is_per_token():
    # no_drop=True (the decode path) processes every token independently: a
    # token's output does not depend on the others sharing the forward.
    torch.manual_seed(0)
    moe = _moe_transformer(grad_checkpoint=False).eval().layers[0].moe
    x = torch.randn(1, 7, 32)
    full = moe(x, no_drop=True)
    single = moe(x[:, 3:4], no_drop=True)  # same token, alone
    assert torch.allclose(full[:, 3:4], single, atol=1e-5)


# ---------------------------------------------------------------------------
# LatentMoE (d_latent set): experts run in a compressed latent space
# ---------------------------------------------------------------------------
def test_latent_moe_only_active_when_d_latent_set():
    plain = MixtureOfExperts(d_model=32, d_hidden=16, n_experts=4)
    assert not hasattr(plain, "weight_compress")
    assert plain.weight_up.shape == (4 * 16, 32)  # experts in d_model

    latent = MixtureOfExperts(d_model=32, d_hidden=16, n_experts=4, d_latent=8)
    assert latent.weight_compress.shape == (8, 32)
    assert latent.weight_expand.shape == (32, 8)
    assert latent.weight_up.shape == (4 * 16, 8)  # experts in d_latent
    assert latent.weight_down.shape == (4 * 8, 16)


def test_latent_moe_forward_shape_and_backward():
    moe = MixtureOfExperts(d_model=32, d_hidden=16, n_experts=4, d_latent=8)
    x = torch.randn(2, 6, 32, requires_grad=True)
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None
    assert moe.weight_compress.grad is not None
    assert moe.weight_expand.grad is not None


def test_latent_moe_decode_matches_full_forward():
    # the latent projections must behave identically on the training path and
    # the per-token decode path (n_active == n_experts so nothing is dropped).
    torch.manual_seed(0)
    m = Transformer(
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_experts=2,
        d_expert=16,
        n_active=2,
        d_latent=8,
        layers_per_block=1,
    ).eval()
    x = torch.randn(1, 6, 32)
    full = m(x)

    out, cache, pos = m.decode(x[:, :4])
    step, _, _ = m.decode(x[:, 4:], cache, pos)
    assert torch.allclose(torch.cat([out, step], dim=1), full, atol=1e-4)


def test_moe_forward_drops_but_no_drop_keeps_every_token():
    # the default forward (training AND validation) keeps the bounded, dropping
    # path -- crucially it must NOT balloon to an n_tokens-sized capacity on a
    # full-batch validation forward. no_drop=True (decode) keeps every token.
    torch.manual_seed(0)
    moe = MixtureOfExperts(
        d_model=8, d_hidden=16, n_experts=4, n_active=1, capacity_factor=1.0
    ).eval()
    with torch.no_grad():
        moe.expert_bias[0] = 1e9  # force every token onto expert 0 -> overflow
    x = torch.randn(1, 20, 8)

    bounded = moe(x)  # capacity = ceil(20/4) = 5 -> 15 tokens dropped (zero out)
    kept = moe(x, no_drop=True)  # capacity = 20 -> nothing dropped
    n_dropped = int((bounded.abs().sum(-1) == 0).sum())
    assert n_dropped > 0  # bounded path drops (train/val behavior)
    assert int((kept.abs().sum(-1) == 0).sum()) == 0  # decode keeps every token
