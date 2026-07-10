import lightning as L
import pytest
import torch

from picochat.model.sft import SFTModule
from picochat.model.transformer import TransformerLM
from picochat.optim import Muon

PAD_ID = 0


def _tiny_lm(**over) -> TransformerLM:
    cfg = dict(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    cfg.update(over)
    return TransformerLM(**cfg)


@pytest.fixture
def sft_module():
    return SFTModule(_tiny_lm(), pad_idx=PAD_ID, compile=False)


def _batch(vocab_size=40, batch_size=2, seq_len=8, mask_last=3):
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    if mask_last:  # last `mask_last` positions are non-trainable
        labels[:, -mask_last:] = PAD_ID
    return {"input_ids": input_ids, "labels": labels}


def test_sft_module_is_lightning_module(sft_module):
    assert isinstance(sft_module, L.LightningModule)
    assert sft_module.pad_idx == PAD_ID


def test_training_step_returns_scalar_loss_and_backprops(sft_module):
    batch = _batch()
    loss = sft_module.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert sft_module.model.embed.weight.grad is not None


def test_validation_step_returns_scalar(sft_module):
    loss = sft_module.validation_step(_batch(), 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_ignores_masked_label_positions():
    # masking a label position (setting it to pad_id) drops it from the loss
    # entirely; unmasking it (giving it a real, mismatched target) must change
    # the loss, proving the position was actually excluded before.
    torch.manual_seed(0)
    lm = _tiny_lm()
    module = SFTModule(lm, pad_idx=PAD_ID, compile=False)
    input_ids = torch.randint(1, 40, (2, 8))
    labels_masked = input_ids.clone()
    labels_masked[:, -3:] = PAD_ID  # last 3 positions masked out

    labels_unmasked = labels_masked.clone()
    labels_unmasked[:, -1] = (input_ids[:, -1] + 1) % 40  # now a real, wrong target

    loss_masked = module._loss(input_ids, labels_masked)
    loss_unmasked = module._loss(input_ids, labels_unmasked)
    assert not torch.allclose(loss_masked, loss_unmasked)


def test_loss_shifts_labels_by_one():
    # an oracle-like setup: labels equal to input_ids means position i's target
    # is input_ids[i+1] after the internal shift, not input_ids[i].
    torch.manual_seed(0)
    lm = _tiny_lm()
    module = SFTModule(lm, pad_idx=PAD_ID, compile=False).eval()  # fixed (no) dropout masks
    x = torch.randint(1, 40, (1, 6))
    loss_shifted = module._loss(x, x.clone())

    logits = lm(x)[:, :-1]
    targets = x[:, 1:]
    import torch.nn.functional as F
    from einops import rearrange

    ref = F.cross_entropy(
        rearrange(logits, "b l v -> (b l) v"),
        rearrange(targets, "b l -> (b l)"),
        ignore_index=PAD_ID,
    )
    assert torch.allclose(loss_shifted, ref)


def test_configure_optimizers_muon(sft_module):
    opt = sft_module.configure_optimizers()
    assert isinstance(opt, Muon)
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_model = sum(p.numel() for p in sft_module.model.parameters())
    assert n_opt == n_model


def test_configure_optimizers_adamw():
    module = SFTModule(_tiny_lm(), pad_idx=PAD_ID, optimizer="adamw", compile=False)
    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)


def test_overfits_single_batch():
    torch.manual_seed(0)
    module = SFTModule(_tiny_lm(), pad_idx=PAD_ID, compile=False)
    opt = torch.optim.Adam(module.parameters(), lr=1e-3)
    batch = _batch(mask_last=0)  # keep every position trainable
    first = module._loss(batch["input_ids"], batch["labels"]).item()
    for _ in range(50):
        opt.zero_grad()
        loss = module._loss(batch["input_ids"], batch["labels"])
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_trainer_fast_dev_run(sft_module):
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self, n=4, seq_len=8):
            self.input_ids = torch.randint(1, 40, (n, seq_len))
            self.labels = self.input_ids.clone()
            self.labels[:, -3:] = PAD_ID

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

    def collate(items):
        return {
            "input_ids": torch.stack([it["input_ids"] for it in items]),
            "labels": torch.stack([it["labels"] for it in items]),
        }

    loader = DataLoader(_DS(), batch_size=2, collate_fn=collate)
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.fit(sft_module, loader, loader)
