import copy

import pytest
import torch

from picochat.gpt import MixtureOfExperts, Transformer, moe_modules


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
    assert plain.bank.weight_up.shape == (4 * 16, 32)  # experts in d_model

    latent = MixtureOfExperts(d_model=32, d_hidden=16, n_experts=4, d_latent=8)
    assert latent.weight_compress.shape == (8, 32)
    assert latent.weight_expand.shape == (32, 8)
    assert latent.bank.weight_up.shape == (4 * 16, 8)  # experts in d_latent
    assert latent.bank.weight_down.shape == (4 * 8, 16)


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


# ---------------------------------------------------------------------------
# share_experts: one routed-expert bank for the whole stack (MoEUT-style)
# ---------------------------------------------------------------------------
def test_share_experts_single_bank_but_per_layer_routing():
    m = Transformer(
        d_model=32,
        n_heads=4,
        n_layers=3,
        n_experts=4,
        d_expert=16,
        layers_per_block=1,
        share_experts=True,
    )
    banks = [layer.moe.bank for layer in m.layers]
    assert all(b is banks[0] for b in banks)  # one shared instance
    # routers and load-balancing biases stay per layer
    assert len({id(layer.moe.weight_router) for layer in m.layers}) == 3
    assert len({id(layer.moe.expert_bias) for layer in m.layers}) == 3
    # parameters() deduplicates: the bank's weights are counted once
    bank_numel = sum(p.numel() for p in banks[0].parameters())
    total = sum(p.numel() for p in m.parameters())
    unshared = _moe_transformer(grad_checkpoint=False)
    assert unshared.layers[0].moe.bank is not unshared.layers[1].moe.bank
    total_unshared = sum(
        p.numel()
        for p in Transformer(
            d_model=32, n_heads=4, n_layers=3, n_experts=4, d_expert=16,
            layers_per_block=1,
        ).parameters()
    )
    assert total == total_unshared - 2 * bank_numel  # 3 banks collapsed to 1


def test_share_experts_bias_updates_stay_per_layer():
    # each layer balances its own routing over the shared pool: one training
    # forward moves every layer's own bias by exactly one rate step.
    torch.manual_seed(0)
    m = Transformer(
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_experts=4,
        d_expert=16,
        layers_per_block=1,
        share_experts=True,
    ).train()
    m(torch.randn(2, 8, 32))
    for layer in m.layers:
        delta = layer.moe.expert_bias
        assert (delta != 0).any()
        assert delta.abs().max() == pytest.approx(layer.moe.bias_update_rate)


def test_share_experts_grads_accumulate_from_every_layer():
    m = Transformer(
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_experts=2,
        d_expert=16,
        n_active=2,
        layers_per_block=1,
        share_experts=True,
    ).train()
    m(torch.randn(2, 6, 32)).sum().backward()
    bank = m.layers[0].moe.bank
    assert bank.weight_up.grad is not None
    assert all(layer.moe.weight_router.grad is not None for layer in m.layers)


def test_share_experts_decode_matches_full_forward_with_latent():
    torch.manual_seed(0)
    m = Transformer(
        d_model=32,
        n_heads=4,
        n_layers=3,
        n_experts=2,
        d_expert=16,
        n_active=2,
        d_latent=8,
        layers_per_block=1,
        share_experts=True,
    ).eval()
    x = torch.randn(1, 6, 32)
    full = m(x)
    out, cache, pos = m.decode(x[:, :4])
    step, _, _ = m.decode(x[:, 4:], cache, pos)
    assert torch.allclose(torch.cat([out, step], dim=1), full, atol=1e-4)


def test_share_experts_state_dict_roundtrip():
    # the shared bank appears under one key per owning layer (same storage);
    # loading those duplicates into a fresh shared model must reproduce it.
    torch.manual_seed(0)

    def build() -> Transformer:
        return Transformer(
            d_model=32,
            n_heads=4,
            n_layers=2,
            n_experts=4,
            d_expert=16,
            layers_per_block=1,
            share_experts=True,
        )

    a, b = build().eval(), build().eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 5, 32)
    assert torch.allclose(a(x), b(x), atol=1e-6)


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


# ---------------------------------------------------------------------------
# expert-output normalization (learnable RMSNorm on the aggregated output)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d_latent", [None, 8])
def test_expert_output_is_rms_normalized(d_latent):
    # every routed (kept) token's aggregated output is RMS-normalized then scaled
    # by out_gain, so with gain=1 each non-zero row has unit RMS regardless of
    # the experts' raw output scale -- this is the stabilizing output norm.
    # (Scale the expert weights up first: at their tiny default init the summed
    # output sits near rms_norm's eps, which is exactly the eps-floored regime
    # that gently ramps the MoE contribution early in training; here we want a
    # healthy magnitude so the assertion tests the normalization itself.)
    torch.manual_seed(0)
    moe = MixtureOfExperts(
        d_model=16, d_hidden=32, n_experts=4, n_active=2, d_latent=d_latent
    )
    assert moe.out_gain.shape == (16,)
    with torch.no_grad():
        for p in moe.parameters():
            if p.ndim >= 2:
                p.mul_(30.0)
    out = moe(torch.randn(1, 8, 16), no_drop=True)  # keep every token
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_out_gain_rescales_output():
    # out_gain is a real, applied scale: doubling it doubles the output RMS.
    torch.manual_seed(0)
    moe = MixtureOfExperts(d_model=16, d_hidden=32, n_experts=4, n_active=2).eval()
    x = torch.randn(1, 8, 16)
    base = moe(x, no_drop=True)
    with torch.no_grad():
        moe.out_gain.mul_(2.0)
    assert torch.allclose(moe(x, no_drop=True), 2.0 * base, atol=1e-4)


# ---------------------------------------------------------------------------
# moe_modules -- discovery of the routed-expert layers in a model
# ---------------------------------------------------------------------------
def test_moe_modules_finds_all_layers():
    t = _moe_transformer(grad_checkpoint=False)  # 2 layers, n_experts=4
    mods = moe_modules(t)
    assert len(mods) == 2
    assert all(m.n_experts == 4 for m in mods)


def test_moe_modules_empty_for_dense_model():
    dense = Transformer(d_model=32, n_heads=4, n_layers=2, layers_per_block=1)
    assert moe_modules(dense) == []
