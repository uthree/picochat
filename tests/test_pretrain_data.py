import lightning as L
import torch
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities.model_helpers import is_overridden
from torch.utils.data import Dataset

from picochat.data.pretrain import PretrainDataModule
from picochat.model.gpt import GPT, TransformerLM


class _RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, n: int):
        self.data = torch.randint(1, vocab_size, (n, seq_len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def test_train_dataloader_uses_batch_size():
    ds = _RandomTokenDataset(40, 6, n=32)
    dm = PretrainDataModule(ds, None, batch_size=4, num_workers=0)
    batch = next(iter(dm.train_dataloader()))
    assert batch.shape[0] == 4


def test_train_dataloader_respects_sample_weights():
    # Two "datasets" of very different sizes; weight them equally so the small
    # one should show up roughly as often as the large one despite having far
    # fewer examples.
    small = _RandomTokenDataset(40, 6, n=4)
    large = _RandomTokenDataset(40, 6, n=400)
    import numpy as np
    from torch.utils.data import ConcatDataset

    ds = ConcatDataset([small, large])
    weights = np.concatenate([np.full(4, 1.0 / 4), np.full(400, 1.0 / 400)])
    dm = PretrainDataModule(ds, None, batch_size=64, num_workers=0, train_sample_weights=weights)

    loader = dm.train_dataloader()
    sampler = loader.sampler
    drawn = list(iter(sampler))
    from_small = sum(1 for i in drawn if i < 4)
    from_large = sum(1 for i in drawn if i >= 4)
    assert from_small > 0
    assert from_large > 0
    # Roughly equal mass given equal weights (loose bound: not proportional to
    # dataset size, which would put ~99% of draws in `large`).
    assert from_small / len(drawn) > 0.2


def test_val_dataloader_uses_batch_size():
    train_ds = _RandomTokenDataset(40, 6, n=32)
    val_ds = _RandomTokenDataset(40, 6, n=32)
    dm = PretrainDataModule(train_ds, val_ds, batch_size=8, num_workers=0)
    batch = next(iter(dm.val_dataloader()))
    assert batch.shape[0] == 8


def test_batch_size_mutation_changes_next_dataloader():
    # Mirrors what Lightning's Tuner does: rewrite `batch_size` in place, then
    # rebuild the dataloader from the new value.
    ds = _RandomTokenDataset(40, 6, n=32)
    dm = PretrainDataModule(ds, None, batch_size=4, num_workers=0)
    dm.batch_size = 16
    batch = next(iter(dm.train_dataloader()))
    assert batch.shape[0] == 16


def test_no_val_dataset_hides_val_dataloader_hook():
    ds = _RandomTokenDataset(40, 6, n=32)
    dm_no_val = PretrainDataModule(ds, None, batch_size=4, num_workers=0)
    assert dm_no_val.val_dataloader is None
    assert not is_overridden("val_dataloader", dm_no_val, parent=L.LightningDataModule)

    dm_with_val = PretrainDataModule(ds, ds, batch_size=4, num_workers=0)
    assert is_overridden("val_dataloader", dm_with_val, parent=L.LightningDataModule)


def _tiny_gpt() -> GPT:
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2, max_seq_len=64)
    return GPT(lm, pad_idx=0, compile=False)


def _tiny_trainer() -> L.Trainer:
    return L.Trainer(
        accelerator="cpu",
        max_steps=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )


def test_scale_batch_size_finds_power_of_two_with_val():
    train_ds = _RandomTokenDataset(40, 6, n=256)
    val_ds = _RandomTokenDataset(40, 6, n=256)
    dm = PretrainDataModule(train_ds, val_ds, batch_size=2, num_workers=0)
    gpt = _tiny_gpt()
    trainer = _tiny_trainer()

    found = Tuner(trainer).scale_batch_size(gpt, datamodule=dm, mode="power", max_val=16)

    assert found is not None
    assert found & (found - 1) == 0  # power of 2
    assert dm.batch_size == found
    # tuner must restore the model/trainer state so the real fit starts clean
    trainer.fit(gpt, datamodule=dm)


def test_scale_batch_size_works_without_val_dataset():
    # Regression test: Lightning's BatchSizeFinder unconditionally probes the
    # val dataloader while restoring state, which crashes if validation_step is
    # defined but no val dataloader exists at all. scripts/base_train.py works
    # around this by shadowing validation_step to None in this scenario.
    train_ds = _RandomTokenDataset(40, 6, n=256)
    dm = PretrainDataModule(train_ds, None, batch_size=2, num_workers=0)
    gpt = _tiny_gpt()
    gpt.validation_step = None
    trainer = _tiny_trainer()

    found = Tuner(trainer).scale_batch_size(gpt, datamodule=dm, mode="power", max_val=16)

    assert found is not None
    assert found & (found - 1) == 0
    trainer.fit(gpt, datamodule=dm)
