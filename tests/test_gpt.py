import pytest
import torch

from picochat.model.gpt import (
    SelfAttention,
    SwiGLU,
    Transformer,
    rms_norm,
    rotate_half,
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


def test_rotate_half_shape_and_involution():
    x = torch.randn(2, 4, 8)
    r = rotate_half(x)
    assert r.shape == x.shape
    # rotating twice negates the original (90-degree rotation applied twice)
    assert torch.allclose(rotate_half(r), -x, atol=1e-5)


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
# SelfAttention
# ---------------------------------------------------------------------------
def test_attention_output_shape():
    attn = SelfAttention(32, 4)
    x = torch.randn(2, 6, 32)
    y, cache = attn(x)
    assert y.shape == x.shape


def test_attention_cache_shape():
    attn = SelfAttention(32, 4)
    x = torch.randn(2, 6, 32)
    _, cache = attn(x)
    # cache stacks [key, value]; each has n_groups heads and seq-len 6
    assert cache.shape == (2, 2, attn.n_groups, 6, attn.d_head)


def test_attention_grouped_query_dims():
    attn = SelfAttention(32, 8, n_groups=2)
    assert attn.n_groups == 2
    assert attn.proj_q.out_features == 32  # 8 heads
    assert attn.proj_k.out_features == attn.d_head * 2  # 2 groups
    assert attn.proj_v.out_features == attn.d_head * 2
    y, _ = attn(torch.randn(2, 4, 32))
    assert y.shape == (2, 4, 32)


def test_attention_invalid_head_division():
    with pytest.raises(AssertionError):
        SelfAttention(30, 4)  # 30 not divisible by 4


def test_attention_invalid_group_division():
    with pytest.raises(AssertionError):
        SelfAttention(32, 8, n_groups=3)  # 8 not divisible by 3


def test_attention_causal_prefix_invariance():
    # causal attention: earlier outputs must not depend on later tokens
    attn = SelfAttention(32, 4).eval()
    x = torch.randn(1, 6, 32)
    full, _ = attn(x)
    prefix, _ = attn(x[:, :3])
    assert torch.allclose(full[:, :3], prefix, atol=1e-5)


def test_attention_cache_matches_full_forward():
    attn = SelfAttention(32, 4).eval()
    x = torch.randn(1, 5, 32)
    full, _ = attn(x)

    # feed first 4 tokens, then the last one using the cache
    _, cache = attn(x[:, :4])
    step, _ = attn(x[:, 4:5], cache=cache)
    assert torch.allclose(full[:, 4:5], step, atol=1e-5)


def test_attention_cache_grows():
    attn = SelfAttention(32, 4).eval()
    _, cache = attn(torch.randn(1, 4, 32))
    _, cache2 = attn(torch.randn(1, 1, 32), cache=cache)
    assert cache2.shape[-2] == 5


def test_attention_backward():
    attn = SelfAttention(32, 4)
    x = torch.randn(2, 5, 32, requires_grad=True)
    y, _ = attn(x)
    y.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
def test_transformer_output_shape():
    model = Transformer(d_model=32, n_heads=4, n_layers=3)
    x = torch.randn(2, 7, 32)
    out, cache = model(x, None)
    assert out.shape == x.shape


def test_transformer_cache_per_layer():
    n_layers = 3
    model = Transformer(d_model=32, n_heads=4, n_layers=n_layers)
    out, cache = model(torch.randn(2, 7, 32), None)
    assert len(cache) == n_layers
    assert all(c is not None for c in cache)


def test_transformer_incremental_matches_full():
    torch.manual_seed(0)
    model = Transformer(d_model=32, n_heads=4, n_layers=2).eval()
    x = torch.randn(1, 5, 32)
    full, _ = model(x, None)

    out, cache = model(x[:, :4], None)
    step, _ = model(x[:, 4:5], cache)
    assert torch.allclose(full[:, 4:5], step, atol=1e-4)


def test_transformer_grouped_query():
    model = Transformer(d_model=32, n_heads=8, n_layers=2, n_groups=2)
    out, cache = model(torch.randn(2, 4, 32), None)
    assert out.shape == (2, 4, 32)
    assert cache[0].shape[2] == 2  # n_groups heads cached


def test_transformer_backward():
    model = Transformer(d_model=32, n_heads=4, n_layers=2)
    x = torch.randn(2, 5, 32, requires_grad=True)
    out, _ = model(x, None)
    out.sum().backward()
    assert x.grad is not None
