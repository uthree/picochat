import torch

from picochat.gpt import (
    DepthAttention,
    SwiGLU,
    Transformer,
    doc_ids_to_cu_seqlens,
    rms_norm,
)


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------
def test_rms_norm_shape_preserved():
    x = torch.randn(2, 3, 8)
    assert rms_norm(x).shape == x.shape


def test_rms_norm_unit_rms():
    x = torch.randn(4, 16) * 5.0
    y = rms_norm(x, eps=0.0)
    rms = y.square().mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rms_norm_scale_invariant():
    x = torch.randn(2, 8)
    a = rms_norm(x, eps=0.0)
    b = rms_norm(x * 10.0, eps=0.0)
    assert torch.allclose(a, b, atol=1e-4)


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------
def test_swiglu_output_shape():
    m = SwiGLU(16).eval()
    x = torch.randn(2, 5, 16)
    assert m(x).shape == x.shape


def test_swiglu_default_hidden_dim():
    m = SwiGLU(16)
    assert m.proj_up.out_features == 16 * 3


def test_swiglu_custom_hidden_dim():
    m = SwiGLU(16, d_hidden=64)
    assert m.proj_up.out_features == 64
    assert m.proj_gate.out_features == 64
    assert m.proj_down.in_features == 64


def test_swiglu_backward():
    m = SwiGLU(16)
    x = torch.randn(2, 5, 16, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_swiglu_eval_is_deterministic():
    m = SwiGLU(16).eval()
    x = torch.randn(2, 5, 16)
    # dropout disabled in eval -> two passes match
    assert torch.allclose(m(x), m(x))


# ---------------------------------------------------------------------------
# DepthAttention (Block AttnRes residual mixing)
# ---------------------------------------------------------------------------
def test_depth_attention_single_source_is_identity():
    # softmax over one source is 1, so a lone block passes through unchanged.
    mix = DepthAttention(16)
    x = torch.randn(1, 5, 16)
    assert torch.allclose(mix(x[None], None), x)


def test_depth_attention_zero_query_mixes_uniformly():
    # the zero-init query gives uniform weights: blocks + partial average.
    mix = DepthAttention(16)
    a, b = torch.randn(2, 1, 5, 16)
    assert torch.allclose(mix(a[None], b), (a + b) / 2, atol=1e-6)


def test_depth_attention_query_selects_sources():
    # a trained (non-zero) query reweights sources per token: push the query
    # far toward one source's direction and the mix approaches that source.
    torch.manual_seed(0)
    mix = DepthAttention(16)
    a, b = torch.randn(2, 1, 5, 16)
    with torch.no_grad():
        mix.query.copy_(100.0 * rms_norm(a)[0, 0])
    out = mix(a[None], b)
    assert torch.allclose(out[:, 0], a[:, 0], atol=1e-3)
    assert not torch.allclose(out, (a + b) / 2, atol=1e-2)


# ---------------------------------------------------------------------------
# Transformer: mixer interleaving (3 GDN : 1 NSA at lpb=4)
# ---------------------------------------------------------------------------
def test_transformer_interleaves_linear_and_sparse_layers():
    # layers_per_block=2 -> every 2nd layer (1-indexed) is NSA (global), the
    # rest are Gated DeltaNet (linear).
    model = Transformer(d_model=32, n_heads=4, n_layers=4, layers_per_block=2)
    kinds = [type(layer.attn).__name__ for layer in model.layers]
    assert kinds == [
        "GatedDeltaNet",
        "NativeSparseAttention",
        "GatedDeltaNet",
        "NativeSparseAttention",
    ]


def test_transformer_default_ratio_is_three_to_one():
    model = Transformer(d_model=32, n_heads=4, n_layers=8, layers_per_block=4)
    linear = [layer.linear for layer in model.layers]
    assert linear == [True, True, True, False, True, True, True, False]


def test_transformer_single_block_is_all_sparse():
    # layers_per_block=1 makes every layer an NSA (global) layer
    model = Transformer(d_model=32, n_heads=4, n_layers=3, layers_per_block=1)
    assert all(not layer.linear for layer in model.layers)


# ---------------------------------------------------------------------------
# packing helper
# ---------------------------------------------------------------------------
def test_doc_ids_to_cu_seqlens_marks_doc_and_row_boundaries():
    doc = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0]])
    cu = doc_ids_to_cu_seqlens(doc)
    # row 0: doc change at flat idx 2; row boundary at 4; row 1 ends at 8
    assert cu.tolist() == [0, 2, 4, 8]


def test_doc_ids_single_doc_per_row():
    doc = torch.tensor([[0, 0, 0], [0, 0, 0]])
    cu = doc_ids_to_cu_seqlens(doc)
    assert cu.tolist() == [0, 3, 6]
