import pytest
import torch

from picochat.model.linear_attn import (
    GatedDeltaNet2,
    chunk_gdn2,
    recurrent_gdn2,
)


# ---------------------------------------------------------------------------
# GDN-2 kernels: chunk == recurrent
# ---------------------------------------------------------------------------
def _rand_inputs(b, t, h, dk, dv, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(b, t, h, dk)
    k = torch.randn(b, t, h, dk)
    v = torch.randn(b, t, h, dv)
    # g is a channel-wise log decay (<= 0); b/w channel-wise gates in (0, 1)
    g = -torch.nn.functional.softplus(torch.randn(b, t, h, dk))
    be = torch.rand(b, t, h, dk)  # erase gate (key axis)
    w = torch.rand(b, t, h, dv)  # write gate (value axis)
    return q, k, v, g, be, w


@pytest.mark.parametrize("t", [1, 7, 64, 130])
def test_chunk_matches_recurrent(t):
    q, k, v, g, be, w = _rand_inputs(2, t, 3, 16, 16)
    out_c, state_c = chunk_gdn2(q, k, v, g, be, w, chunk_size=64)
    out_r, state_r = recurrent_gdn2(q, k, v, g, be, w)
    assert out_c.shape == (2, t, 3, 16)
    assert torch.allclose(out_c, out_r, atol=1e-4)
    assert torch.allclose(state_c, state_r, atol=1e-4)


def test_recurrent_state_carries_across_calls():
    # splitting a sequence and threading the state must equal one call
    q, k, v, g, be, w = _rand_inputs(1, 10, 2, 8, 8)
    full, _ = recurrent_gdn2(q, k, v, g, be, w)
    a, state = recurrent_gdn2(
        q[:, :6], k[:, :6], v[:, :6], g[:, :6], be[:, :6], w[:, :6]
    )
    b, _ = recurrent_gdn2(
        q[:, 6:],
        k[:, 6:],
        v[:, 6:],
        g[:, 6:],
        be[:, 6:],
        w[:, 6:],
        initial_state=state,
    )
    assert torch.allclose(torch.cat([a, b], dim=1), full, atol=1e-4)


def test_erase_and_write_gates_decoupled():
    # GDN-2's defining property vs Gated DeltaNet: with the write gate w == 0
    # nothing is committed, but the erase side still edits the state.
    q, k, v, g, be, w = _rand_inputs(1, 6, 2, 8, 8)
    state0 = torch.randn(1, 2, 8, 8)
    _, state_no_write = recurrent_gdn2(
        q, k, v, g, be, torch.zeros_like(w), initial_state=state0
    )
    _, state_frozen = recurrent_gdn2(
        q,
        k,
        v,
        torch.zeros_like(g),
        torch.zeros_like(be),
        torch.zeros_like(w),
        initial_state=state0,
    )
    # no decay, no erase, no write -> the state is exactly carried through
    assert torch.allclose(state_frozen, state0.float(), atol=1e-5)
    # erase/decay active with zero write -> the state still changes
    assert not torch.allclose(state_no_write, state0.float(), atol=1e-3)


# ---------------------------------------------------------------------------
# GatedDeltaNet2 module
# ---------------------------------------------------------------------------
def test_gdn_output_shape():
    m = GatedDeltaNet2(32, n_heads=4).eval()
    x = torch.randn(2, 6, 32)
    assert m(x).shape == x.shape


def test_gdn_grouped_heads():
    m = GatedDeltaNet2(32, n_heads=8, n_kv_heads=2)
    assert m.key_dim == 2 * (32 // 8)
    assert m.value_dim == 32
    assert m(torch.randn(2, 5, 32)).shape == (2, 5, 32)


def test_gdn_causal_prefix_invariance():
    m = GatedDeltaNet2(32, n_heads=4).eval()
    x = torch.randn(1, 8, 32)
    full = m(x)
    prefix = m(x[:, :4])
    assert torch.allclose(full[:, :4], prefix, atol=1e-4)


def test_gdn_decode_matches_forward():
    torch.manual_seed(0)
    m = GatedDeltaNet2(32, n_heads=4).eval()
    x = torch.randn(1, 6, 32)
    full = m(x)
    # prefill first 4, then step the remaining tokens one at a time
    out0, state = m.decode(x[:, :4])
    outs = [out0]
    for t in range(4, 6):
        o, state = m.decode(x[:, t : t + 1], state)
        outs.append(o)
    decoded = torch.cat(outs, dim=1)
    assert torch.allclose(decoded, full, atol=1e-4)


def test_gdn_decode_single_steps_match_forward():
    torch.manual_seed(1)
    m = GatedDeltaNet2(24, n_heads=3).eval()
    x = torch.randn(1, 5, 24)
    full = m(x)
    state, outs = None, []
    for t in range(5):
        o, state = m.decode(x[:, t : t + 1], state)
        outs.append(o)
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=1e-4)


def test_gdn_cu_seqlens_resets_state():
    # with cu_seqlens splitting the row into two docs, the second doc must not
    # depend on the first (state reset at the boundary)
    torch.manual_seed(0)
    m = GatedDeltaNet2(16, n_heads=2).eval()
    x = torch.randn(1, 8, 16)
    cu = torch.tensor([0, 4, 8])
    out = m(x, cu_seqlens=cu)

    perturbed = x.clone()
    perturbed[:, 0] += 100.0
    out2 = m(perturbed, cu_seqlens=cu)
    # first doc changes, second doc (positions 4..7) is unaffected
    assert not torch.allclose(out[:, :4], out2[:, :4], atol=1e-4)
    assert torch.allclose(out[:, 4:], out2[:, 4:], atol=1e-4)


def test_gdn_backward():
    m = GatedDeltaNet2(32, n_heads=4)
    x = torch.randn(2, 5, 32, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_chunk_backward_finite_under_strong_decay():
    # With strong decays the anticausal pairwise exponents overflow to inf if
    # they are exp'd before masking, which poisons the backward with NaNs (the
    # forward stays finite -- only the gradients break). Regression test for
    # the exp-then-mask pitfall.
    q, k, v, g, be, w = _rand_inputs(1, 40, 2, 8, 8)
    g = g * 30
    q.requires_grad_(True)
    out, _ = chunk_gdn2(q, k, v, g, be, w, chunk_size=16)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(q.grad).all()
