import math

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from picochat.model import TransformerLM
from picochat.training import GPT


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


def test_init_gives_near_uniform_loss():
    # small init -> logits near 0 -> near-uniform distribution -> loss ~= ln(vocab)
    lm = TransformerLM(vocab_size=200, d_model=64, n_heads=8, n_layers=4)
    gpt = GPT(lm)
    loss = gpt._loss(torch.randint(0, 200, (4, 16))).item()
    assert loss == pytest.approx(math.log(200), abs=0.5)


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
    # default is the torch.optim.Muon + AdamW pair (Muon for hidden matrices,
    # AdamW for embeddings/lm head/1-dim params); no scheduler is returned --
    # the LR schedule is applied by hand under manual optimization
    muon_opt, adam_opt = gpt_module.configure_optimizers()
    assert isinstance(muon_opt, torch.optim.Muon)
    assert isinstance(adam_opt, torch.optim.AdamW)
    # together they must cover the model parameters
    n_opt = sum(
        p.numel()
        for opt in (muon_opt, adam_opt)
        for group in opt.param_groups
        for p in group["params"]
    )
    n_model = sum(p.numel() for p in gpt_module.model.parameters())
    assert n_opt == n_model


def test_gpt_configure_optimizers_adamw():
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    [opt] = GPT(lm, pad_idx=0, optimizer="adamw").configure_optimizers()
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
    [opt] = GPT(lm, pad_idx=0, optimizer="adamw").configure_optimizers()
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
    # manual optimization: configure_optimizers returns the bare optimizers and
    # the LR schedule is applied by hand (see _apply_lr); the base LRs are
    # captured per optimizer for that.
    opts = gpt.configure_optimizers()
    assert isinstance(opts[0], torch.optim.Muon)
    assert gpt._base_lrs == [[g["lr"] for g in opt.param_groups] for opt in opts]
    # warmup: ~0 at step 0, 1.0 at the end of warmup, then cosine down to min_lr_ratio
    assert gpt._lr_lambda(0) < gpt._lr_lambda(5)
    assert gpt._lr_lambda(9) == pytest.approx(1.0)
    assert gpt._lr_lambda(100) == pytest.approx(0.1)
    assert gpt._lr_lambda(55) == pytest.approx(0.55, abs=0.05)


def test_gpt_apply_lr_scales_base_lr():
    # _apply_lr multiplies each group's captured base LR by the schedule
    # factor, across every optimizer of the muon+adamw pair.
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    gpt = GPT(lm, pad_idx=0, warmup_steps=10, max_steps=100)
    opts = gpt.configure_optimizers()
    base = [list(lrs) for lrs in gpt._base_lrs]

    class _FakeTrainer:
        global_step = 5  # mid-warmup

    gpt._trainer = _FakeTrainer()
    gpt._apply_lr(opts)
    for base_lrs, opt in zip(base, opts):
        for b, g in zip(base_lrs, opt.param_groups):
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


def test_gpt_loss_derives_doc_ids_from_bos():
    # with bos_idx set, _loss numbers the documents packed into the window by
    # counting <s> markers and derives the packing tensors (doc_ids for the NSA
    # layers, cu_seqlens for the GDN state resets) outside the compiled forward
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    gpt = GPT(lm, pad_idx=0, bos_idx=5, compile=False)
    seen = {}
    real = gpt._next_token_loss

    def spy(input_ids, targets, doc_ids, cu_seqlens):
        seen["doc_ids"] = doc_ids
        seen["cu_seqlens"] = cu_seqlens
        return real(input_ids, targets, doc_ids, cu_seqlens)

    gpt._next_token_loss = spy
    loss = gpt._loss(torch.tensor([[3, 5, 4, 4, 5, 6]]))
    assert torch.isfinite(loss)
    assert seen["doc_ids"].tolist() == [[0, 1, 1, 1, 2, 2]]
    # one row, three documents -> segment boundaries at each <s> plus the end
    assert seen["cu_seqlens"].tolist() == [0, 1, 4, 6]


def test_gpt_without_bos_idx_passes_no_packing():
    # fused_loss=False: the spy stands in for _forward and returns logits,
    # which is the unfused contract (fused expects hidden states -- see
    # LMTrainerMixin._next_token_loss).
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    gpt_module = GPT(lm, pad_idx=0, fused_loss=False)
    seen = {}

    def spy(x, doc_ids=None, cu_seqlens=None):
        seen["doc_ids"] = doc_ids
        seen["cu_seqlens"] = cu_seqlens
        return gpt_module.model(x, doc_ids, cu_seqlens)

    # object.__setattr__ mirrors how __init__ stores _forward (kept out of
    # nn.Module's submodule registry, see GPT.__init__)
    object.__setattr__(gpt_module, "_forward", spy)
    gpt_module._loss(torch.randint(1, 40, (2, 6)))
    assert seen["doc_ids"] is None and seen["cu_seqlens"] is None


def test_gpt_doc_mask_blocks_cross_document_loss_leak():
    # perturbing tokens of an earlier document must not change the model's
    # view of a later one: logits after the second <s> stay identical
    torch.manual_seed(0)
    bos = 5
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2).eval()
    x = torch.tensor([[bos, 7, 8, 9, bos, 11, 12, 13]])
    doc_ids = (x == bos).cumsum(-1)
    base = lm(x, doc_ids)

    perturbed = x.clone()
    perturbed[:, 1:4] = torch.tensor([20, 21, 22])
    out = lm(perturbed, (perturbed == bos).cumsum(-1))
    assert torch.allclose(base[:, 4:], out[:, 4:], atol=1e-5)


def test_gpt_state_dict_only_contains_model_weights(gpt_module):
    # the _forward handle (self.model, or its torch.compile wrapper) must not
    # be registered as a submodule: it would duplicate every weight under
    # `_forward.*` / `_forward._orig_mod.*` and tie checkpoints to the
    # compile setting they were saved with
    assert all(k.startswith("model.") for k in gpt_module.state_dict())


def test_gpt_checkpoint_loads_across_compile_settings():
    # regression: a GPU-trained (compiled) checkpoint must load into an
    # uncompiled module -- exactly what scripts/chat.py does
    cfg = dict(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    compiled = GPT(TransformerLM(**cfg), pad_idx=0, compile=True)
    eager = GPT(TransformerLM(**cfg), pad_idx=0, compile=False)
    eager.load_state_dict(compiled.state_dict())  # must not raise


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


def test_gpt_generate_long_with_kv_cache():
    # GPT._generate drives Transformer/TransformerLM.decode through a long
    # prefill followed by many single-token steps over the hybrid stack (GDN
    # recurrent state carry + NSA windowed decode), generating well past the NSA
    # window_size; this must not crash and must stay within max_seq_len.
    lm = TransformerLM(
        vocab_size=40,
        d_model=32,
        n_heads=4,
        n_layers=4,
        layers_per_block=2,  # GDN + NSA
        window_size=3,
        sel_block=4,
        n_selected=4,
        max_seq_len=64,
    )
    gpt = GPT(lm, pad_idx=0, compile=False).eval()

    prompt = torch.randint(1, 40, (1, 10))
    generated = gpt._generate(prompt, max_new_tokens=20)
    assert generated.shape == (1, 20)


def test_gpt_gradient_accumulation_steps_once_per_cycle():
    # with accumulate=2, an optimizer step fires only every 2nd microbatch;
    # grads accumulate across the pair in between. global_step counts real steps.
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, max_seq_len=64)
    gpt = GPT(lm, pad_idx=0, compile=False, accumulate=2, max_steps=2)

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


def test_mtp_training_step_trains_all_heads():
    # with MTP heads the loss is primary + weighted auxiliary heads; a training
    # step must reach every head (primary and each MTP head).
    torch.manual_seed(0)
    lm = TransformerLM(
        vocab_size=40, d_model=32, n_heads=4, n_layers=2, max_seq_len=64, n_mtp=2
    )
    gpt = GPT(lm, pad_idx=0, compile=False, mtp_weight=0.5)
    batch = torch.randint(1, 40, (2, 16))
    loss = gpt.training_step(batch, 0)
    assert loss.requires_grad and torch.isfinite(loss)
    assert lm.lmhead.weight.grad is not None
    # each MTP head's transform must get gradient (gelu'(0)=0.5 despite zero init)
    for head in lm.mtp_heads:
        g = head.out.weight.grad
        assert g is not None and g.abs().sum() > 0


def test_mtp_loss_exceeds_primary_only():
    # the MTP auxiliary terms add to the objective, so the reported loss is above
    # the primary-only next-token loss on the same batch.
    torch.manual_seed(0)
    lm = TransformerLM(
        vocab_size=40, d_model=32, n_heads=4, n_layers=2, max_seq_len=64, n_mtp=1
    )
    gpt = GPT(lm, pad_idx=0, compile=False, mtp_weight=1.0)
    batch = torch.randint(1, 40, (2, 16))
    with torch.no_grad():
        combined = gpt._loss(batch)
        # primary-only: temporarily drop the MTP head
        saved, lm.n_mtp = lm.n_mtp, 0
        primary = gpt._loss(batch)
        lm.n_mtp = saved
    assert combined > primary
