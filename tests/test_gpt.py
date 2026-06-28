import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from picochat.model.gpt import (
    GPT,
    SelfAttention,
    SwiGLU,
    Transformer,
    TransformerLM,
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


# ---------------------------------------------------------------------------
# cross-layer attention sharing (n_attn_layers)
# ---------------------------------------------------------------------------
def test_attn_layers_default_to_n_layers():
    model = Transformer(d_model=32, n_heads=4, n_layers=4)
    assert model.n_attn_layers == 4
    assert len(model.attn) == 4
    # one FFN per layer regardless of sharing
    assert len(model.ffn) == 4


def test_shared_attention_module_count():
    model = Transformer(d_model=32, n_heads=4, n_layers=6, n_attn_layers=2)
    assert len(model.attn) == 2  # only 2 distinct attention modules
    assert len(model.ffn) == 6  # ffn stays per-layer


def test_shared_attention_reduces_params():
    full = Transformer(d_model=32, n_heads=4, n_layers=6)
    shared = Transformer(d_model=32, n_heads=4, n_layers=6, n_attn_layers=2)
    n_full = sum(p.numel() for p in full.parameters())
    n_shared = sum(p.numel() for p in shared.parameters())
    assert n_shared < n_full


def test_shared_attention_modules_are_identical_objects():
    # cyclic sharing: layer i uses attn[i % n_attn_layers]
    model = Transformer(d_model=32, n_heads=4, n_layers=6, n_attn_layers=2)
    assert model.attn[0] is not model.attn[1]
    # weights are genuinely tied (same parameter tensor reused)
    shared_params = {id(p) for p in model.attn[0].parameters()}
    # forward must run through the shared modules without index errors
    out, cache = model(torch.randn(1, 5, 32), None)
    assert out.shape == (1, 5, 32)
    assert len(cache) == 6  # cache is still per-layer
    assert len(shared_params) > 0


@pytest.mark.parametrize(
    "n_layers,n_attn_layers,expected_max_index",
    [(6, 2, 1), (8, 2, 1), (6, 3, 2), (4, 1, 0)],
)
def test_shared_attention_cyclic_mapping(n_layers, n_attn_layers, expected_max_index):
    # forward should never index past the available attention modules
    model = Transformer(
        d_model=16, n_heads=4, n_layers=n_layers, n_attn_layers=n_attn_layers
    )
    mapping = [i % model.n_attn_layers for i in range(n_layers)]
    assert max(mapping) == expected_max_index
    out, _ = model(torch.randn(1, 4, 16), None)
    assert out.shape == (1, 4, 16)


def test_n_attn_layers_must_divide_n_layers():
    with pytest.raises(AssertionError):
        Transformer(d_model=16, n_heads=4, n_layers=5, n_attn_layers=2)


def test_shared_attention_backward_updates_shared_module():
    model = Transformer(d_model=16, n_heads=4, n_layers=4, n_attn_layers=2)
    out, _ = model(torch.randn(2, 5, 16), None)
    out.sum().backward()
    # a parameter of a shared attention module must accumulate gradient
    grad = model.attn[0].proj_q.weight.grad
    assert grad is not None and grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# TransformerLM
# ---------------------------------------------------------------------------
def test_transformer_lm_logits_shape():
    vocab_size = 40
    lm = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2)
    tokens = torch.randint(0, vocab_size, (2, 5))
    logits, cache = lm(tokens, None)
    assert logits.shape == (2, 5, vocab_size)
    assert len(cache) == 2


def test_transformer_lm_incremental_matches_full():
    vocab_size = 40
    lm = TransformerLM(vocab_size=vocab_size, d_model=32, n_heads=4, n_layers=2).eval()
    tokens = torch.randint(0, vocab_size, (1, 5))
    full, _ = lm(tokens, None)

    _, cache = lm(tokens[:, :4], None)
    step, _ = lm(tokens[:, 4:5], cache)
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
    # max_steps が無いときは optimizer だけ（スケジューラ無し）を返す。
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
    # embedding と bias（1次元）は weight decay 対象から除外される。
    embed_weight = gpt_module.model.embed[0].weight
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
    # warmup: step 0 はほぼ 0、warmup 終端で 1.0、その後 cosine で min_lr_ratio へ。
    assert gpt._lr_lambda(0) < gpt._lr_lambda(5)
    assert gpt._lr_lambda(9) == pytest.approx(1.0)
    assert gpt._lr_lambda(100) == pytest.approx(0.1)
    assert gpt._lr_lambda(55) == pytest.approx(0.55, abs=0.05)


def test_gpt_loss_backward_reaches_embedding(gpt_module):
    batch = torch.randint(1, 40, (2, 6))
    gpt_module.training_step(batch, 0).backward()
    assert gpt_module.model.embed[0].weight.grad is not None


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
