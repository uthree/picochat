import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from picochat.gpt import (
    DepthAttention,
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
    y = attn(x)
    assert y.shape == x.shape


def test_attention_cache_shape():
    attn = SelfAttention(32, 4)
    x = torch.randn(2, 6, 32)
    _, cache = attn.decode(x)
    # cache stacks [key, value]; each has n_kv_heads heads and seq-len 6
    assert cache.shape == (2, 2, attn.n_kv_heads, 6, attn.d_head)


def test_attention_grouped_query_dims():
    # d_model=32, n_heads=8 -> d_head=4; 2 KV heads.
    attn = SelfAttention(32, 8, n_kv_heads=2)
    assert attn.n_heads == 8
    assert attn.n_kv_heads == 2
    assert attn.d_head == 4  # d_model // n_heads
    assert attn.proj_q.out_features == 4 * 8  # d_head * n_heads == d_model
    assert attn.proj_k.out_features == 4 * 2  # d_head * n_kv_heads
    assert attn.proj_v.out_features == 4 * 2
    assert attn.proj_o.in_features == 4 * 8
    y = attn(torch.randn(2, 4, 32))
    assert y.shape == (2, 4, 32)


def test_d_head_derived_from_n_heads():
    # per-head dim is d_model // n_heads; proj_q stays square.
    attn = SelfAttention(64, 4)
    assert attn.n_heads == 4
    assert attn.d_head == 16  # 64 // 4
    assert attn.n_kv_heads == 4  # defaults to MHA
    assert attn.proj_q.out_features == 64  # square


def test_attention_invalid_head_division():
    with pytest.raises(AssertionError):
        SelfAttention(32, 5)  # 32 not divisible by 5


def test_attention_invalid_d_head_odd():
    with pytest.raises(AssertionError):
        SelfAttention(30, 2)  # d_head=15 is odd; RoPE needs an even d_head


def test_attention_invalid_group_division():
    # d_model=32, n_heads=4; 3 KV heads does not divide 4.
    with pytest.raises(AssertionError):
        SelfAttention(32, 4, n_kv_heads=3)


def test_attention_causal_prefix_invariance():
    # causal attention: earlier outputs must not depend on later tokens
    attn = SelfAttention(32, 4).eval()
    x = torch.randn(1, 6, 32)
    full = attn(x)
    prefix = attn(x[:, :3])
    assert torch.allclose(full[:, :3], prefix, atol=1e-5)


def test_attention_cache_matches_full_forward():
    attn = SelfAttention(32, 4).eval()
    x = torch.randn(1, 5, 32)
    full = attn(x)

    # feed first 4 tokens, then the last one using the cache
    _, cache = attn.decode(x[:, :4])
    step, _ = attn.decode(x[:, 4:5], cache=cache, pos=4)
    assert torch.allclose(full[:, 4:5], step, atol=1e-5)


def test_attention_cache_grows():
    attn = SelfAttention(32, 4).eval()
    _, cache = attn.decode(torch.randn(1, 4, 32))
    _, cache2 = attn.decode(torch.randn(1, 1, 32), cache=cache, pos=4)
    assert cache2.shape[-2] == 5


def test_attention_backward():
    attn = SelfAttention(32, 4)
    x = torch.randn(2, 5, 32, requires_grad=True)
    y = attn(x)
    y.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# Sliding window attention
# ---------------------------------------------------------------------------
def test_window_attention_output_shape():
    attn = SelfAttention(32, 4, window_size=3)
    x = torch.randn(2, 10, 32)
    assert attn(x).shape == x.shape


def test_window_attention_ignores_tokens_outside_window():
    attn = SelfAttention(32, 4, window_size=3).eval()
    x = torch.randn(1, 10, 32)
    full = attn(x)

    perturbed = x.clone()
    perturbed[:, 0] += 100.0
    out = attn(perturbed)

    # position 9 is 9 steps away from position 0, well outside the window of 3
    assert torch.allclose(full[:, 9], out[:, 9], atol=1e-5)
    # position 1 is within the window, so it must be affected
    assert not torch.allclose(full[:, 1], out[:, 1], atol=1e-5)


def test_window_attention_still_causal():
    # earlier outputs must not depend on later tokens, window or not
    attn = SelfAttention(32, 4, window_size=3).eval()
    x = torch.randn(1, 6, 32)
    full = attn(x)
    prefix = attn(x[:, :3])
    assert torch.allclose(full[:, :3], prefix, atol=1e-5)


def test_window_attention_cache_matches_full_forward():
    attn = SelfAttention(32, 4, window_size=3).eval()
    x = torch.randn(1, 10, 32)
    full = attn(x)

    _, cache = attn.decode(x[:, :8])
    step, _ = attn.decode(x[:, 8:9], cache=cache, pos=8)
    assert torch.allclose(full[:, 8:9], step, atol=1e-4)


def test_window_attention_cache_is_bounded():
    # the KV cache for a windowed layer must never grow past window_size,
    # regardless of how many tokens have been decoded so far.
    window = 3
    attn = SelfAttention(32, 4, window_size=window).eval()
    cache, pos = None, 0
    for _ in range(10):
        x = torch.randn(1, 1, 32)
        _, cache = attn.decode(x, cache=cache, pos=pos)
        pos += x.shape[1]
        assert cache.shape[-2] <= window


def test_window_attention_long_prefill_exceeds_window():
    # a single decode() call whose chunk is longer than window_size (e.g. the
    # prefill call in GPT._generate) must still match a from-scratch forward():
    # truncating the KV cache must not corrupt the attention computed for the
    # tokens *within* that same over-long chunk.
    window = 3
    attn = SelfAttention(32, 4, window_size=window).eval()
    x = torch.randn(1, 12, 32)
    full = attn(x)

    cache, pos = None, 0
    prefill_len = 7  # > window_size
    out_prefill, cache = attn.decode(x[:, :prefill_len], cache=cache, pos=pos)
    pos += prefill_len
    assert cache.shape[-2] == window

    outs = [out_prefill]
    for t in range(prefill_len, x.shape[1]):
        out, cache = attn.decode(x[:, t : t + 1], cache=cache, pos=pos)
        pos += 1
        outs.append(out)
        assert cache.shape[-2] <= window
    decoded = torch.cat(outs, dim=1)
    assert torch.allclose(decoded, full, atol=1e-4)


def test_window_attention_mixed_chunk_sizes():
    # several chunk sizes in one decode session, some larger than window_size,
    # must all agree with a from-scratch forward() over the same tokens.
    window = 3
    attn = SelfAttention(32, 4, window_size=window).eval()
    x = torch.randn(1, 15, 32)
    full = attn(x)

    cache, pos = None, 0
    outs = []
    for chunk_len in (5, 4, 1, 5):
        chunk = x[:, pos : pos + chunk_len]
        out, cache = attn.decode(chunk, cache=cache, pos=pos)
        pos += chunk_len
        outs.append(out)
        assert cache.shape[-2] <= window
    decoded = torch.cat(outs, dim=1)
    assert torch.allclose(decoded, full, atol=1e-4)


def test_window_none_means_full_attention():
    # window_size=None should be equivalent to unbounded (full) attention
    torch.manual_seed(0)
    attn = SelfAttention(32, 4, window_size=None).eval()
    x = torch.randn(1, 10, 32)
    full = attn(x)

    perturbed = x.clone()
    perturbed[:, 0] += 100.0
    out = attn(perturbed)
    assert not torch.allclose(full[:, 9], out[:, 9], atol=1e-5)


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


def test_transformer_interleaves_global_and_local_layers():
    # layers_per_block=2 -> every 2nd layer (1-indexed) is full attention,
    # the rest are windowed.
    model = Transformer(
        d_model=32, n_heads=4, n_layers=4, window_size=3, layers_per_block=2
    )
    window_sizes = [layer.attn.window_size for layer in model.layers]
    assert window_sizes == [3, None, 3, None]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flex_attention path is CUDA-only"
)
def test_window_attention_flex_attention_matches_masked_sdpa_on_cuda():
    # On CUDA, forward() routes windowed attention through flex_attention
    # instead of a materialized SDPA mask; verify the two give the same result.
    torch.manual_seed(0)
    attn = SelfAttention(64, 8, window_size=3).cuda().eval()
    x = torch.randn(2, 12, 64, device="cuda")

    flex_out = attn(x)

    query, key, value = attn._project(x)
    query, key = attn._rope(query), attn._rope(key)
    mask = attn._window_mask(
        query.shape[-2], key.shape[-2], q_offset=0, k_offset=0, device=query.device
    )
    ref = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, enable_gqa=True
    )
    ref_out = attn.proj_o(rearrange(ref, "b h l d -> b l (h d)"))

    assert torch.allclose(flex_out, ref_out, atol=1e-3)


# ---------------------------------------------------------------------------
# RoPE max_seq_len (decoupled from rope_base)
# ---------------------------------------------------------------------------
def test_rope_table_sized_by_max_seq_len():
    attn = SelfAttention(32, 4, max_seq_len=128)
    # table length is max_seq_len, not rope_base (10000)
    assert attn.sin.shape[0] == 128
    assert attn.cos.shape[0] == 128


def test_rope_tables_not_in_state_dict():
    # derived buffers must not bloat checkpoints / break loading on resize
    keys = SelfAttention(32, 4, max_seq_len=128).state_dict().keys()
    assert not any(k.endswith("sin") or k.endswith("cos") for k in keys)


def test_rope_allows_context_beyond_10000():
    # the old code capped positions at rope_base=10000; now it is configurable
    attn = SelfAttention(16, 2, max_seq_len=12000).eval()
    x = torch.randn(1, 11000, 16)
    out = attn(x)
    assert out.shape == (1, 11000, 16)


def test_rope_raises_past_max_seq_len():
    attn = SelfAttention(32, 4, max_seq_len=16).eval()
    with pytest.raises(AssertionError):
        attn(torch.randn(1, 17, 32))
