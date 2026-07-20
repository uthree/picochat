import torch

from picochat.sparse_attn import NativeSparseAttention, PartialRoPE


def _nsa(d_model=32, n_heads=4, **kw):
    # small blocks so tests exercise compression/selection on short sequences
    defaults = dict(
        cmp_block=4, cmp_stride=2, sel_block=4, n_selected=4, window=4, max_seq_len=256
    )
    defaults.update(kw)
    return NativeSparseAttention(d_model, n_heads, **defaults)


# ---------------------------------------------------------------------------
# PartialRoPE
# ---------------------------------------------------------------------------
def test_partial_rope_rotates_only_prefix():
    rope = PartialRoPE(d_head=16, factor=0.25, base=1e6, max_seq_len=32)
    assert rope.rot_dim == 4  # round(16*0.25)=4, even
    x = torch.randn(1, 2, 5, 16)
    y = rope(x, torch.arange(5))
    # un-rotated tail is unchanged
    assert torch.allclose(y[..., 4:], x[..., 4:])
    assert not torch.allclose(y[..., :4], x[..., :4])


def test_partial_rope_zero_factor_is_identity():
    rope = PartialRoPE(d_head=16, factor=0.0, base=1e6, max_seq_len=32)
    assert rope.rot_dim == 0
    x = torch.randn(1, 2, 5, 16)
    assert torch.allclose(rope(x, torch.arange(5)), x)


# ---------------------------------------------------------------------------
# NSA forward
# ---------------------------------------------------------------------------
def test_nsa_output_shape():
    m = _nsa().eval()
    x = torch.randn(2, 20, 32)
    assert m(x).shape == x.shape


def test_nsa_grouped_query():
    m = _nsa(n_heads=8, n_kv_heads=2).eval()
    assert m(torch.randn(2, 16, 32)).shape == (2, 16, 32)


def test_nsa_short_sequence_window_only():
    # sequence shorter than a compression block: only the window branch fires
    m = _nsa(cmp_block=8, cmp_stride=4).eval()
    x = torch.randn(1, 3, 32)
    assert m(x).shape == (1, 3, 32)


def test_nsa_causal_prefix_invariance():
    m = _nsa().eval()
    x = torch.randn(1, 24, 32)
    full = m(x)
    prefix = m(x[:, :10])
    assert torch.allclose(full[:, :10], prefix, atol=1e-4)


def test_nsa_later_token_does_not_affect_earlier():
    m = _nsa().eval()
    x = torch.randn(1, 24, 32)
    full = m(x)
    perturbed = x.clone()
    perturbed[:, 20:] += 50.0
    out = m(perturbed)
    assert torch.allclose(full[:, :12], out[:, :12], atol=1e-4)


def test_nsa_decode_matches_forward():
    torch.manual_seed(0)
    m = _nsa().eval()
    x = torch.randn(1, 18, 32)
    full = m(x)
    out0, cache = m.decode(x[:, :10], pos=0)
    outs = [out0]
    for t in range(10, 18):
        o, cache = m.decode(x[:, t : t + 1], cache, pos=t)
        outs.append(o)
    decoded = torch.cat(outs, dim=1)
    assert torch.allclose(decoded, full, atol=1e-4)


def test_nsa_decode_step_by_step():
    torch.manual_seed(1)
    m = _nsa(n_heads=4).eval()
    x = torch.randn(1, 15, 32)
    full = m(x)
    cache, outs = None, []
    for t in range(15):
        o, cache = m.decode(x[:, t : t + 1], cache, pos=t)
        outs.append(o)
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=1e-4)


def test_nsa_doc_masking_isolates_documents():
    # two packed docs in one row: perturbing doc 0 must not change doc 1 outputs
    torch.manual_seed(0)
    m = _nsa().eval()
    x = torch.randn(1, 20, 32)
    doc = torch.tensor([[0] * 10 + [1] * 10])
    out = m(x, doc_ids=doc)
    perturbed = x.clone()
    perturbed[:, :10] += 50.0
    out2 = m(perturbed, doc_ids=doc)
    assert torch.allclose(out[:, 10:], out2[:, 10:], atol=1e-4)
    assert not torch.allclose(out[:, :10], out2[:, :10], atol=1e-4)


def test_nsa_backward_reaches_all_branches():
    m = _nsa()
    x = torch.randn(1, 20, 32, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None
    # compression / gate / selection projections all receive gradient
    assert m.phi_k.weight.grad is not None
    assert m.gate.weight.grad is not None
    assert m.proj_k_slc.weight.grad is not None
    assert m.proj_k_win.weight.grad is not None
