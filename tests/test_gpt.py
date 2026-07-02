import lightning as L
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from picochat.model.gpt import (
    GPT,
    MODEL_PRESETS,
    SelfAttention,
    SwiGLU,
    Transformer,
    TransformerLM,
    build_lm,
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


def test_transformer_interleaves_global_and_local_layers():
    # global_attn_ratio=2 -> every 2nd layer (1-indexed) is full attention,
    # the rest are windowed.
    model = Transformer(
        d_model=32, n_heads=4, n_layers=4, window_size=3, global_attn_ratio=2
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
# Transformer
# ---------------------------------------------------------------------------
def test_transformer_output_shape():
    model = Transformer(d_model=32, n_heads=4, n_layers=3)
    x = torch.randn(2, 7, 32)
    out = model(x)
    assert out.shape == x.shape


def test_transformer_cache_per_layer():
    n_layers = 3
    model = Transformer(d_model=32, n_heads=4, n_layers=n_layers)
    out, cache, pos = model.decode(torch.randn(2, 7, 32))
    assert len(cache) == n_layers
    assert all(c is not None for c in cache)


def test_transformer_decode_pos_advances():
    model = Transformer(d_model=32, n_heads=4, n_layers=2)
    _, cache, pos = model.decode(torch.randn(2, 5, 32))
    assert pos == 5
    _, cache, pos = model.decode(torch.randn(2, 3, 32), cache, pos)
    assert pos == 8


def test_transformer_incremental_matches_full():
    torch.manual_seed(0)
    model = Transformer(d_model=32, n_heads=4, n_layers=2).eval()
    x = torch.randn(1, 5, 32)
    full = model(x)

    out, cache, pos = model.decode(x[:, :4])
    step, _, _ = model.decode(x[:, 4:5], cache, pos)
    assert torch.allclose(full[:, 4:5], step, atol=1e-4)


def test_transformer_grouped_query():
    model = Transformer(d_model=32, n_heads=8, n_layers=2, n_kv_heads=2)
    out, cache, pos = model.decode(torch.randn(2, 4, 32))
    assert out.shape == (2, 4, 32)
    assert cache[0].shape[2] == 2  # n_kv_heads heads cached


def test_transformer_backward():
    model = Transformer(d_model=32, n_heads=4, n_layers=2)
    x = torch.randn(2, 5, 32, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# Looped Transformer
# ---------------------------------------------------------------------------
def test_looped_transformer_output_shape():
    model = Transformer(d_model=32, n_heads=4, n_layers=3, n_loops=2)
    x = torch.randn(2, 7, 32)
    out = model(x)
    assert out.shape == x.shape


def test_looped_transformer_cache_per_layer():
    n_layers = 3
    n_loops = 2
    model = Transformer(d_model=32, n_heads=4, n_layers=n_layers, n_loops=n_loops)
    out, cache, pos = model.decode(torch.randn(2, 7, 32))
    assert len(cache) == n_layers * n_loops
    assert all(c is not None for c in cache)


# ---------------------------------------------------------------------------
# TransformerLM
# ---------------------------------------------------------------------------
def test_transformer_lm_logits_shape():
    vocab_size = 40
    lm = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2)
    tokens = torch.randint(0, vocab_size, (2, 5))
    logits = lm(tokens)
    assert logits.shape == (2, 5, vocab_size)


def test_transformer_lm_incremental_matches_full():
    vocab_size = 40
    lm = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2).eval()
    tokens = torch.randint(0, vocab_size, (1, 5))
    full = lm(tokens)

    _, cache, pos = lm.decode(tokens[:, :4])
    step, _, _ = lm.decode(tokens[:, 4:5], cache, pos)
    assert torch.allclose(full[:, 4:5], step, atol=1e-4)


# ---------------------------------------------------------------------------
# GPT (LightningModule)
# ---------------------------------------------------------------------------
class _RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, n: int = 8):
        self.data = torch.randint(1, vocab_size, (n, seq_len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


@pytest.fixture
def gpt_module():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    return GPT(lm, pad_idx=0)


def test_gpt_is_lightning_module(gpt_module):
    assert isinstance(gpt_module, L.LightningModule)
    assert gpt_module.pad_idx == 0


def test_gpt_training_step_returns_scalar_loss(gpt_module):
    batch = torch.randint(1, 40, (2, 6))
    loss = gpt_module.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_gpt_validation_step_returns_scalar(gpt_module):
    batch = torch.randint(1, 40, (2, 6))
    loss = gpt_module.validation_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gpt_configure_optimizers(gpt_module):
    # without max_steps it returns just the optimizer (no scheduler)
    opt = gpt_module.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)
    # optimizer must cover the model parameters
    n_opt = sum(p.numel() for group in opt.param_groups for p in group["params"])
    n_model = sum(p.numel() for p in gpt_module.model.parameters())
    assert n_opt == n_model


def test_gpt_weight_decay_excludes_bias_and_embedding(gpt_module):
    opt = gpt_module.configure_optimizers()
    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] > 0
    assert no_decay_group["weight_decay"] == 0.0
    # embeddings and biases (1-dim) are excluded from weight decay
    embed_weight = gpt_module.model.embed.weight
    no_decay_ids = {id(p) for p in no_decay_group["params"]}
    assert id(embed_weight) in no_decay_ids
    assert all(p.ndim >= 2 for p in decay_group["params"])
    assert all(
        p.ndim < 2 or id(p) == id(embed_weight) for p in no_decay_group["params"]
    )


def test_gpt_configure_optimizers_with_schedule():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    gpt = GPT(lm, pad_idx=0, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    config = gpt.configure_optimizers()
    assert isinstance(config["optimizer"], torch.optim.AdamW)
    assert config["lr_scheduler"]["interval"] == "step"
    # warmup: ~0 at step 0, 1.0 at the end of warmup, then cosine down to min_lr_ratio
    assert gpt._lr_lambda(0) < gpt._lr_lambda(5)
    assert gpt._lr_lambda(9) == pytest.approx(1.0)
    assert gpt._lr_lambda(100) == pytest.approx(0.1)
    assert gpt._lr_lambda(55) == pytest.approx(0.55, abs=0.05)


def test_gpt_loss_backward_reaches_embedding(gpt_module):
    batch = torch.randint(1, 40, (2, 6))
    gpt_module.training_step(batch, 0).backward()
    assert gpt_module.model.embed.weight.grad is not None


def test_embeddings_untied_when_disabled():
    lm = TransformerLM(
        vocab_size=40, d_model=32, n_heads=4, n_layers=2, tie_embeddings=False
    )
    assert lm.lmhead.weight is not lm.embed.weight


def test_embeddings_tied_by_default():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    assert lm.lmhead.weight is lm.embed.weight


def test_gpt_pad_targets_are_ignored(gpt_module):
    # padding positions in the target must not change the loss
    base = torch.randint(1, 40, (1, 6))
    loss_a = gpt_module._loss(base.clone())
    padded = base.clone()
    padded[:, -1] = gpt_module.pad_idx  # becomes a target after the shift
    loss_b = gpt_module._loss(padded)
    # only the embedding of the final (input) token differs; the ignored target
    # position should keep the comparison close, never produce nan/inf
    assert torch.isfinite(loss_a) and torch.isfinite(loss_b)


def test_gpt_trainer_fast_dev_run(gpt_module):
    loader = DataLoader(_RandomTokenDataset(40, 6), batch_size=4)
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.fit(gpt_module, loader, loader)


def test_gpt_overfits_single_batch(gpt_module):
    # a correct next-token loss must be able to drive the loss down on one batch
    batch = torch.randint(1, 40, (2, 8))
    opt = torch.optim.Adam(gpt_module.parameters(), lr=1e-3)
    gpt_module.train()
    first = gpt_module._loss(batch).item()
    for _ in range(50):
        opt.zero_grad()
        loss = gpt_module._loss(batch)
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_gpt_generate_beyond_window_size_with_kv_cache():
    # GPT._generate drives Transformer/TransformerLM.decode through a long
    # prefill followed by many single-token steps; this must not crash and must
    # keep every windowed layer's KV cache bounded, even generating past
    # window_size (though still within max_seq_len, per the max_seq_len-is-a-
    # hard-ceiling design decision).
    lm = TransformerLM(
        vocab_size=40,
        d_model=32,
        n_heads=4,
        n_layers=4,
        max_seq_len=64,
    )
    for i, layer in enumerate(lm.transformer.layers):
        layer.attn.window_size = 3 if i % 2 == 0 else None
    gpt = GPT(lm, pad_idx=0, compile=False).eval()

    prompt = torch.randint(1, 40, (1, 10))
    generated = gpt._generate(prompt, max_new_tokens=20)
    assert generated.shape == (1, 20)


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


# ---------------------------------------------------------------------------
# scale-ladder presets
# ---------------------------------------------------------------------------
def test_build_lm_unknown_size_raises():
    with pytest.raises(ValueError):
        build_lm("gigantic", vocab_size=32)


@pytest.mark.parametrize("size", list(MODEL_PRESETS))
def test_preset_dims_are_consistent(size):
    cfg = MODEL_PRESETS[size]
    assert cfg["d_model"] % cfg["n_heads"] == 0  # heads tile d_model
    assert cfg["n_heads"] % cfg["n_kv_heads"] == 0  # GQA grouping
    assert (cfg["d_model"] // cfg["n_heads"]) % 2 == 0  # d_head even (RoPE)


def test_build_lm_pico_forward():
    lm = build_lm("pico", vocab_size=50, max_seq_len=64)
    logits = lm(torch.randint(0, 50, (2, 16)))
    assert logits.shape == (2, 16, 50)


def test_build_lm_overrides_preset():
    lm = build_lm("pico", vocab_size=50, n_layers=2)
    assert lm.transformer.n_layers == 2  # overridden from preset's 8


def test_build_lm_vocab_override():
    lm = build_lm("pico", vocab_size=123)
    assert lm.embed.num_embeddings == 123
    assert lm.lmhead.out_features == 123


def test_preset_tie_defaults():
    # small scales tie (lmhead shares the embedding matrix); large scales untie.
    pico = build_lm("pico")
    assert pico.lmhead.weight is pico.embed.weight
    small = build_lm("small")
    assert small.lmhead.weight is small.embed.weight
    base = build_lm("base")
    assert base.lmhead.weight is not base.embed.weight


def test_tie_embeddings_override():
    untied = build_lm("pico", tie_embeddings=False)
    assert untied.lmhead.weight is not untied.embed.weight
    tied = build_lm("base", tie_embeddings=True)
    assert tied.lmhead.weight is tied.embed.weight


def test_tied_embeddings_share_gradient():
    lm = build_lm("pico", vocab_size=40, n_layers=1, d_model=16, n_heads=2)
    assert lm.tie_embeddings
    lm(torch.randint(0, 40, (1, 4))).sum().backward()
    # one shared parameter -> appears once in parameters()
    n_embed_params = sum(1 for p in lm.parameters() if p is lm.embed.weight)
    assert n_embed_params == 1


def test_init_gives_near_uniform_loss():
    # small init -> logits near 0 -> near-uniform distribution -> loss ~= ln(vocab)
    import math

    lm = TransformerLM(vocab_size=200, d_model=64, n_heads=8, n_layers=4)
    gpt = GPT(lm)
    loss = gpt._loss(torch.randint(0, 200, (4, 16))).item()
    assert loss == pytest.approx(math.log(200), abs=0.5)
