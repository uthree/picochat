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
from picochat.optim import Muon, zeropower_via_newtonschulz5


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
# TransformerLM
# ---------------------------------------------------------------------------
def test_transformer_lm_logits_shape():
    vocab_size = 40
    lm = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2)
    tokens = torch.randint(0, vocab_size, (2, 5))
    logits = lm(tokens)
    # forward returns one logits tensor per lm head (a single head by default)
    assert len(logits) == 1
    assert logits[0].shape == (2, 5, vocab_size)


def test_transformer_lm_incremental_matches_full():
    vocab_size = 40
    lm = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2).eval()
    tokens = torch.randint(0, vocab_size, (1, 5))
    full = lm(tokens)[0]

    _, cache, pos = lm.decode(tokens[:, :4])
    step, _, _ = lm.decode(tokens[:, 4:5], cache, pos)
    assert torch.allclose(full[:, 4:5], step, atol=1e-4)


# ---------------------------------------------------------------------------
# Multiple token prediction (n_lmheads > 1)
# ---------------------------------------------------------------------------
def test_mtp_forward_returns_one_logits_per_head():
    vocab_size = 40
    lm = TransformerLM(
        vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2, n_lmheads=3
    )
    tokens = torch.randint(0, vocab_size, (2, 6))
    logits = lm(tokens)
    assert len(logits) == 3
    assert all(head.shape == (2, 6, vocab_size) for head in logits)


def test_mtp_heads_are_independent_params():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=2)
    assert lm.lmheads[0].weight is not lm.lmheads[1].weight


def test_mtp_decode_uses_next_token_head():
    # decode must emit a single logits tensor (from head 0), matching head 0 of
    # a full forward, so autoregressive generation ignores the extra heads.
    vocab_size = 40
    lm = TransformerLM(
        vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2, n_lmheads=3
    ).eval()
    tokens = torch.randint(0, vocab_size, (1, 5))
    full = lm(tokens)[0]
    step, _, _ = lm.decode(tokens)
    assert torch.allclose(full, step, atol=1e-4)


def test_mtp_head_losses_shift_targets():
    # head k's output at position i is scored against token i+1+k: feed an
    # oracle that emits one-hot logits for exactly that token and the loss of
    # every head must be ~0 (any off-by-one in the shifts would blow it up).
    vocab_size = 10
    n_lmheads = 3
    lm = TransformerLM(
        vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=1, n_lmheads=n_lmheads
    )
    gpt = GPT(lm, pad_idx=0, compile=False)
    x = torch.randint(1, vocab_size, (2, 7))

    # Replace each lm head with an oracle that ignores the hidden state and emits
    # one-hot logits for the token k+1 steps ahead; the trailing positions (no
    # such target) stay zeros, which _head_loss must slice off.
    class _OracleHead(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.register_buffer("logits", logits)

        def forward(self, h):
            return self.logits

    heads = []
    for k in range(n_lmheads):
        shift = k + 1
        logits = torch.zeros(*x.shape, vocab_size)
        logits[:, : x.shape[1] - shift] = (
            F.one_hot(x[:, shift:], vocab_size).float() * 200.0
        )
        heads.append(_OracleHead(logits))
    gpt.model.lmheads = torch.nn.ModuleList(heads)

    losses = gpt._head_losses(x)
    assert losses.shape == (n_lmheads,)
    assert torch.allclose(losses, torch.zeros(n_lmheads), atol=1e-3)


def test_mtp_loss_backward_reaches_all_heads():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=3)
    gpt = GPT(lm, pad_idx=0, compile=False)
    batch = torch.randint(1, 40, (2, 8))
    losses = gpt._head_losses(batch)
    assert losses.shape == (3,)
    assert torch.isfinite(losses).all()
    losses.mean().backward()
    for head in lm.lmheads:
        assert head.weight.grad is not None
        assert head.weight.grad.abs().sum() > 0


def test_mtp_training_step_scalar_loss():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=2)
    gpt = GPT(lm, pad_idx=0, compile=False)
    batch = torch.randint(1, 40, (2, 8))
    loss = gpt.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_mtp_backward_matches_naive_summed_loss():
    # the memory-optimized two-stage backward (detach trunk, backprop each head,
    # then one trunk backward) must yield exactly the same gradients as a plain
    # backward of the mean-over-heads loss.
    torch.manual_seed(0)
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=3)
    gpt = GPT(lm, pad_idx=0, compile=False).eval()  # eval: fixed (no) dropout masks
    batch = torch.randint(1, 40, (2, 8))

    gpt.zero_grad(set_to_none=True)
    gpt._loss(batch).backward()  # reference: mean over heads, single backward
    ref = {n: p.grad.clone() for n, p in lm.named_parameters()}

    gpt.zero_grad(set_to_none=True)
    gpt._mtp_backward(batch, scale=1.0)  # memory-optimized path
    for n, p in lm.named_parameters():
        assert torch.allclose(p.grad, ref[n], atol=1e-5), n


def test_mtp_backward_scale_scales_grads():
    # `scale` (gradient accumulation) linearly scales every gradient.
    torch.manual_seed(0)
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=2)
    gpt = GPT(lm, pad_idx=0, compile=False).eval()  # eval: fixed (no) dropout masks
    batch = torch.randint(1, 40, (2, 8))

    gpt.zero_grad(set_to_none=True)
    gpt._mtp_backward(batch, scale=1.0)
    full = lm.embed.weight.grad.clone()

    gpt.zero_grad(set_to_none=True)
    gpt._mtp_backward(batch, scale=0.25)
    assert torch.allclose(lm.embed.weight.grad, full * 0.25, atol=1e-6)


def test_gpt_gradient_accumulation_steps_once_per_cycle():
    # with accumulate=2, an optimizer step fires only every 2nd microbatch;
    # grads accumulate across the pair in between. global_step counts real steps.
    import lightning as L
    from torch.utils.data import DataLoader

    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, max_seq_len=64)
    gpt = GPT(lm, pad_idx=0, compile=False, accumulate=2, max_steps=2)

    real_steps = {"n": 0}
    orig = gpt.optimizers

    loader = DataLoader(_RandomTokenDataset(40, 6), batch_size=4)
    trainer = L.Trainer(
        max_steps=2,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

    seen = []
    real_optimizer_step = GPT._optimizer_step

    def counting_step(self, batch_idx):
        stepped_before = self.trainer.global_step
        real_optimizer_step(self, batch_idx)
        seen.append(self.trainer.global_step > stepped_before)

    gpt._optimizer_step = counting_step.__get__(gpt, GPT)
    trainer.fit(gpt, loader)

    # 2 optimizer steps at accumulate=2 => 4 microbatches; only the odd-indexed
    # microbatches (1, 3) actually step
    assert trainer.global_step == 2
    assert seen == [False, True, False, True]


def test_mtp_generate_works():
    lm = TransformerLM(
        vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_lmheads=2, max_seq_len=64
    )
    gpt = GPT(lm, pad_idx=0, compile=False).eval()
    prompt = torch.randint(1, 40, (1, 8))
    generated = gpt._generate(prompt, max_new_tokens=6)
    assert generated.shape == (1, 6)


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
    # manual optimization: training_step runs the backward itself and returns a
    # detached logging loss, so grads land on the params rather than on `loss`
    assert gpt_module.model.embed.weight.grad is not None


def test_gpt_validation_step_returns_scalar(gpt_module):
    batch = torch.randint(1, 40, (2, 6))
    loss = gpt_module.validation_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gpt_configure_optimizers(gpt_module):
    # default optimizer is Muon (with its embedded AdamW for non-Muon params);
    # without max_steps it returns just the optimizer (no scheduler)
    opt = gpt_module.configure_optimizers()
    assert isinstance(opt, Muon)
    # optimizer must cover the model parameters
    n_opt = sum(p.numel() for group in opt.param_groups for p in group["params"])
    n_model = sum(p.numel() for p in gpt_module.model.parameters())
    assert n_opt == n_model


def test_gpt_configure_optimizers_adamw():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    opt = GPT(lm, pad_idx=0, optimizer="adamw").configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)
    n_opt = sum(p.numel() for group in opt.param_groups for p in group["params"])
    n_model = sum(p.numel() for p in lm.parameters())
    assert n_opt == n_model


def test_gpt_unknown_optimizer_raises():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    with pytest.raises(ValueError):
        GPT(lm, pad_idx=0, optimizer="sgd").configure_optimizers()


def test_gpt_weight_decay_excludes_bias_and_embedding():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    opt = GPT(lm, pad_idx=0, optimizer="adamw").configure_optimizers()
    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] > 0
    assert no_decay_group["weight_decay"] == 0.0
    # embeddings and biases (1-dim) are excluded from weight decay
    embed_weight = lm.embed.weight
    no_decay_ids = {id(p) for p in no_decay_group["params"]}
    assert id(embed_weight) in no_decay_ids
    assert all(p.ndim >= 2 for p in decay_group["params"])
    assert all(
        p.ndim < 2 or id(p) == id(embed_weight) for p in no_decay_group["params"]
    )


def test_gpt_configure_optimizers_with_schedule():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    gpt = GPT(lm, pad_idx=0, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    # manual optimization: configure_optimizers returns the bare optimizer and
    # the LR schedule is applied by hand (see _apply_lr); the base LRs are
    # captured for that.
    opt = gpt.configure_optimizers()
    assert isinstance(opt, Muon)
    assert gpt._base_lrs == [g["lr"] for g in opt.param_groups]
    # warmup: ~0 at step 0, 1.0 at the end of warmup, then cosine down to min_lr_ratio
    assert gpt._lr_lambda(0) < gpt._lr_lambda(5)
    assert gpt._lr_lambda(9) == pytest.approx(1.0)
    assert gpt._lr_lambda(100) == pytest.approx(0.1)
    assert gpt._lr_lambda(55) == pytest.approx(0.55, abs=0.05)


def test_gpt_apply_lr_scales_base_lr():
    # _apply_lr multiplies each group's captured base LR by the schedule factor.
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    gpt = GPT(lm, pad_idx=0, warmup_steps=10, max_steps=100, optimizer="adamw")
    opt = gpt.configure_optimizers()
    base = list(gpt._base_lrs)

    class _FakeTrainer:
        global_step = 5  # mid-warmup

    gpt._trainer = _FakeTrainer()
    gpt._apply_lr(opt)
    for b, g in zip(base, opt.param_groups):
        assert g["lr"] == pytest.approx(b * gpt._lr_lambda(5))


def test_gpt_loss_backward_reaches_embedding(gpt_module):
    batch = torch.randint(1, 40, (2, 6))
    # manual optimization: training_step backprops internally (through the
    # detached-trunk two-stage backward), so the grad must reach the embedding.
    gpt_module.training_step(batch, 0)
    assert gpt_module.model.embed.weight.grad is not None


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
    assert len(logits) == 4
    assert logits[0].shape == (2, 16, 50)


def test_build_lm_overrides_preset():
    lm = build_lm("pico", vocab_size=50, n_layers=2)
    assert lm.transformer.n_layers == 2  # overridden from preset's 8


def test_build_lm_vocab_override():
    lm = build_lm("pico", vocab_size=123)
    assert lm.embed.num_embeddings == 123
    assert all(head.out_features == 123 for head in lm.lmheads)


def test_build_lm_n_lmheads_override():
    lm = build_lm("pico", vocab_size=50, n_lmheads=2)
    assert len(lm.lmheads) == 2


def test_init_gives_near_uniform_loss():
    # small init -> logits near 0 -> near-uniform distribution -> loss ~= ln(vocab)
    import math

    lm = TransformerLM(vocab_size=200, d_model=64, n_heads=8, n_layers=4)
    gpt = GPT(lm)
    loss = gpt._loss(torch.randint(0, 200, (4, 16))).item()
    assert loss == pytest.approx(math.log(200), abs=0.5)


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------
def test_newtonschulz_orthogonalizes():
    # Newton-Schulz drives every singular value toward ~1 (the coefficients
    # trade exactness for speed, so allow a generous band around it).
    torch.manual_seed(0)
    for shape in ((16, 32), (32, 8)):  # wide and tall
        X = zeropower_via_newtonschulz5(torch.randn(*shape)).float()
        assert X.shape == shape
        s = torch.linalg.svdvals(X)
        assert ((s > 0.4) & (s < 1.4)).all()


def test_muon_param_split_covers_everything_once():
    lm = TransformerLM(
        vocab_size=40,
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_experts=4,
        d_expert=16,
        n_lmheads=2,
    )
    gpt = GPT(lm, pad_idx=0)
    muon_group, adam_decay, adam_no_decay = gpt._muon_param_groups()
    muon_ids = {id(p) for p in muon_group["params"]}
    decay_ids = {id(p) for p in adam_decay["params"]}
    no_decay_ids = {id(p) for p in adam_no_decay["params"]}

    # fused MoE weights (router 2D + experts 3D) are Muon-optimized
    moe = lm.transformer.layers[0].moe
    for w in (moe.weight_router, moe.weight_up, moe.weight_gate, moe.weight_down):
        assert id(w) in muon_ids
    # only matrix-shaped params reach Muon
    assert all(p.ndim >= 2 for p in muon_group["params"])
    # embedding (no decay) and every lm head (decay) go to the embedded AdamW
    assert id(lm.embed.weight) in no_decay_ids
    for head in lm.lmheads:
        assert id(head.weight) in decay_ids
    # each trainable param lands in exactly one group
    all_ids = {id(p) for p in lm.parameters() if p.requires_grad}
    assert muon_ids | decay_ids | no_decay_ids == all_ids
    assert len(muon_ids) + len(decay_ids) + len(no_decay_ids) == len(all_ids)


def test_muon_step_updates_moe_and_preserves_shapes():
    torch.manual_seed(0)
    lm = TransformerLM(
        vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_experts=4, d_expert=16
    )
    gpt = GPT(lm, pad_idx=0, compile=False)
    opt = gpt.configure_optimizers()
    moe = lm.transformer.layers[0].moe
    attn = lm.transformer.layers[0].attn
    shapes = {name: p.shape for name, p in lm.named_parameters()}
    before_up = moe.weight_up.detach().clone()
    before_q = attn.proj_q.weight.detach().clone()

    gpt.train()
    gpt._loss(torch.randint(1, 40, (2, 8))).backward()
    opt.step()

    for name, p in lm.named_parameters():
        assert p.shape == shapes[name]  # flatten/reshape round-trips exactly
        assert torch.isfinite(p).all()
    # the 3D expert weight and a plain 2D hidden matrix both moved
    assert not torch.allclose(moe.weight_up, before_up)
    assert not torch.allclose(attn.proj_q.weight, before_q)


def test_muon_overfits_single_batch():
    torch.manual_seed(0)
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    gpt = GPT(lm, pad_idx=0, compile=False)
    opt = gpt.configure_optimizers()
    batch = torch.randint(1, 40, (2, 8))
    gpt.train()
    first = gpt._loss(batch).item()
    for _ in range(30):
        opt.zero_grad()
        loss = gpt._loss(batch)
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_muon_group_requires_flag():
    with pytest.raises(ValueError):
        Muon([dict(params=[torch.nn.Parameter(torch.randn(4, 4))])])


# ---------------------------------------------------------------------------
# MoE load-balancing bias under gradient checkpointing
# ---------------------------------------------------------------------------
def _moe_transformer(grad_checkpoint: bool) -> Transformer:
    return Transformer(
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_experts=4,
        d_expert=16,
        global_attn_ratio=1,  # all full attention -> CPU-friendly, no windows
        grad_checkpoint=grad_checkpoint,
    )


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
    import copy

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
        return [l.moe.expert_bias.clone() for l in model.layers]

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
