import pytest
import torch

from picochat.sparse_attn import _HAS_FLA, NativeSparseAttention, PartialRoPE


def _nsa(d_model=32, n_heads=4, **kw):
    # small blocks so tests exercise compression/selection on short sequences
    defaults = dict(
        n_kv_heads=1, block_size=4, n_selected=4, window=4, max_seq_len=256
    )
    defaults.update(kw)
    return NativeSparseAttention(d_model, n_heads, **defaults)


# ---------------------------------------------------------------------------
# PartialRoPE
# ---------------------------------------------------------------------------
def test_partial_rope_rotates_only_prefix():
    rope = PartialRoPE(d_head=16, factor=0.25, base=1e6, max_seq_len=32)
    assert rope.rot_dim == 4  # round(16*0.25)=4, even
    x = torch.randn(1, 5, 2, 16)  # (b, l, h, d)
    y = rope(x, torch.arange(5))
    # un-rotated tail is unchanged
    assert torch.allclose(y[..., 4:], x[..., 4:])
    assert not torch.allclose(y[..., :4], x[..., :4])


def test_partial_rope_zero_factor_is_identity():
    rope = PartialRoPE(d_head=16, factor=0.0, base=1e6, max_seq_len=32)
    assert rope.rot_dim == 0
    x = torch.randn(1, 5, 2, 16)
    assert torch.allclose(rope(x, torch.arange(5)), x)


# ---------------------------------------------------------------------------
# NSA forward (pure-PyTorch reference path)
# ---------------------------------------------------------------------------
def test_nsa_output_shape():
    m = _nsa().eval()
    x = torch.randn(2, 20, 32)
    assert m(x).shape == x.shape


def test_nsa_grouped_query():
    m = _nsa(n_heads=8, n_kv_heads=2).eval()
    assert m(torch.randn(2, 16, 32)).shape == (2, 16, 32)


def test_nsa_short_sequence():
    # sequence shorter than one block: no complete compression block exists;
    # the forced (current-block) selection + window still cover the tokens
    m = _nsa(block_size=8).eval()
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


def test_nsa_cu_seqlens_isolates_documents():
    # two packed docs in one row: perturbing doc 0 must not change doc 1
    # outputs (each cu_seqlens segment runs as its own sequence)
    torch.manual_seed(0)
    m = _nsa().eval()
    x = torch.randn(1, 20, 32)
    cu = torch.tensor([0, 10, 20])
    out = m(x, cu_seqlens=cu)
    perturbed = x.clone()
    perturbed[:, :10] += 50.0
    out2 = m(perturbed, cu_seqlens=cu)
    assert torch.allclose(out[:, 10:], out2[:, 10:], atol=1e-4)
    assert not torch.allclose(out[:, :10], out2[:, :10], atol=1e-4)


def test_nsa_selection_subset_runs():
    # n_selected smaller than the number of candidate blocks: the top-k path
    # actually discards blocks and still produces causal, finite outputs
    torch.manual_seed(0)
    m = _nsa(block_size=4, n_selected=3, window=4).eval()
    x = torch.randn(1, 40, 32)
    out = m(x)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_nsa_backward_reaches_all_projections():
    m = _nsa()
    x = torch.randn(1, 20, 32, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None
    for p in (m.proj_q, m.proj_k, m.proj_v, m.proj_o, m.gate):
        assert p.weight.grad is not None


# ---------------------------------------------------------------------------
# CUDA: fla Triton kernel path vs the reference path
# ---------------------------------------------------------------------------
_CUDA_OK = _HAS_FLA and torch.cuda.is_available()


def _kernel_nsa():
    # group size 16 (16 q heads, MQA) as the fla kernels require
    return NativeSparseAttention(
        256, 16, n_kv_heads=1, block_size=64, n_selected=4, window=128,
        max_seq_len=1024,
    )


@pytest.mark.skipif(not _CUDA_OK, reason="needs CUDA + fla")
def test_nsa_kernel_matches_reference_cuda():
    torch.manual_seed(0)
    m = _kernel_nsa().cuda().eval()
    x = torch.randn(2, 512, 256, device="cuda")
    assert m._use_kernel(x)
    with torch.no_grad():
        out_kernel = m(x)
        m._use_kernel = lambda x: False  # force the reference path
        out_ref = m(x)
    assert torch.allclose(out_kernel, out_ref, atol=2e-3), (
        (out_kernel - out_ref).abs().max().item()
    )


@pytest.mark.skipif(not _CUDA_OK, reason="needs CUDA + fla")
def test_nsa_kernel_matches_reference_cuda_packed():
    torch.manual_seed(0)
    m = _kernel_nsa().cuda().eval()
    x = torch.randn(1, 512, 256, device="cuda")
    cu = torch.tensor([0, 200, 512], device="cuda", dtype=torch.long)
    with torch.no_grad():
        out_kernel = m(x, cu_seqlens=cu)
        m._use_kernel = lambda x: False
        out_ref = m(x, cu_seqlens=cu)
    assert torch.allclose(out_kernel, out_ref, atol=2e-3), (
        (out_kernel - out_ref).abs().max().item()
    )


@pytest.mark.skipif(not _CUDA_OK, reason="needs CUDA + fla")
def test_nsa_kernel_backward_cuda():
    torch.manual_seed(0)
    m = _kernel_nsa().cuda()
    x = torch.randn(1, 256, 256, device="cuda", requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None
    for p in (m.proj_q, m.proj_k, m.proj_v, m.proj_o, m.gate):
        assert p.weight.grad is not None
