"""Dataset that reads the flat token binary produced by scripts/base_setup.py.

The file is a continuous token stream concatenated without padding. Slicing a
block_size+1 window and returning it lets GPT._loss shift by one internally to
compute the next-token prediction loss (sequence length block_size+1 ->
effective context block_size).
"""

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

# 32-bit token ids: leaves headroom for vocab beyond 65535 (e.g. up to 128k).
# Writer (scripts/base_setup.py) imports this so the two never diverge.
DTYPE = np.uint32


class PackedDataset(Dataset):
    def __init__(self, path: str, block_size: int = 1024, random: bool = True):
        """
        Args:
            path: the .bin file produced by base_setup.py
            block_size: effective context length. Each sample is block_size+1 tokens.
            random: True for random offsets, False for non-overlapping contiguous
                blocks.
        """
        self.path = path
        self.block_size = block_size
        self.random = random
        # Determine the length up front. The memmap itself is opened after the
        # worker fork (see below).
        n = np.memmap(path, dtype=DTYPE, mode="r").shape[0]
        self.n_tokens = int(n)
        self._data: np.memmap | None = None
        assert self.n_tokens > block_size, (
            f"corpus ({self.n_tokens} tokens) is shorter than "
            f"block_size+1 ({block_size + 1})"
        )

    @property
    def data(self) -> np.memmap:
        # Reopen per DataLoader worker process: holding the memmap from __init__
        # can break because the file descriptor would be shared across the fork.
        if self._data is None:
            self._data = np.memmap(self.path, dtype=DTYPE, mode="r")
        return self._data

    def __len__(self) -> int:
        if self.random:
            return self.n_tokens - self.block_size
        return self.n_tokens // (self.block_size + 1)

    def __getitem__(self, idx: int) -> Tensor:
        if self.random:
            start = idx
        else:
            start = idx * (self.block_size + 1)
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        return torch.from_numpy(chunk)


class GroupWeightedIndexSampler(Sampler[int]):
    """Draws indices from a ConcatDataset's groups with replacement so each
    group's total sampling mass matches its configured weight, regardless of
    the group's size.

    `torch.utils.data.WeightedRandomSampler` would do this by materializing
    one weight per example and drawing via `torch.multinomial`, which (a)
    refuses more than 2**24 (~16.7M) categories -- easily exceeded once
    several pretraining corpora are concatenated -- and (b) at the scale of
    billions of examples, a fp32/fp64 cumulative-weight table loses enough
    precision that many adjacent examples become unreachable. Sampling in two
    stages -- pick a group via a tiny multinomial (one category per group),
    then a uniform offset within it -- needs O(num_groups) memory and has
    neither problem.
    """

    def __init__(self, group_sizes: list[int], group_weights: list[float], num_samples: int):
        self.num_samples = num_samples
        self.group_weights = torch.as_tensor(group_weights, dtype=torch.double)
        sizes = torch.as_tensor(group_sizes, dtype=torch.long)
        self.group_sizes = sizes
        self.group_offsets = torch.cumsum(sizes, dim=0) - sizes

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        group_ids = torch.multinomial(self.group_weights, self.num_samples, replacement=True)
        local = (torch.rand(self.num_samples, dtype=torch.double) * self.group_sizes[group_ids]).long()
        idx = self.group_offsets[group_ids] + local
        yield from idx.tolist()


class PretrainDataModule(L.LightningDataModule):
    """Wraps train/val datasets with a plain `batch_size` attribute.

    Lightning's Tuner rewrites `batch_size` in place and rebuilds the
    dataloaders from it (see scripts/base_train.py's auto batch-size search), so
    the dataloaders must be built from this attribute rather than fixed at
    construction time.
    """

    def __init__(
        self,
        train_ds: Dataset,
        val_ds: Dataset | None,
        batch_size: int,
        num_workers: int = 4,
        train_group_weights: list[float] | None = None,
    ):
        """
        Args:
            train_group_weights: one weight per group in `train_ds` (which
                must be a ConcatDataset of those groups), sized so each
                group's total sampling mass equals its configured weight
                regardless of its example count. None -> plain uniform
                shuffling.
        """
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_group_weights = train_group_weights
        if train_group_weights is not None:
            assert isinstance(train_ds, ConcatDataset), (
                "train_group_weights requires train_ds to be a ConcatDataset"
            )
        if val_ds is None:
            # Shadow the class method: Lightning's is_overridden() check treats
            # an instance attribute of None as "hook not provided".
            self.val_dataloader = None  # type: ignore[assignment]

    def train_dataloader(self) -> DataLoader:
        if self.train_group_weights is not None:
            group_sizes = [len(d) for d in self.train_ds.datasets]
            sampler = GroupWeightedIndexSampler(
                group_sizes,
                self.train_group_weights,
                num_samples=len(self.train_ds),
            )
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                pin_memory=True,
                drop_last=True,
            )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )
