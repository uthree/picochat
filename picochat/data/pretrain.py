"""Sharded token binaries: the writer used by scripts/base_setup.py and the
dataset that reads them back.

A corpus is a continuous token stream concatenated without padding, split
across fixed-size shard files (00000.bin, 00001.bin, ...) in one directory so
no single file grows with the corpus. Slicing a block_size+1 window and
returning it lets GPT._loss shift by one internally to compute the next-token
prediction loss (sequence length block_size+1 -> effective context block_size).
"""

from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

# 32-bit token ids: leaves headroom for vocab beyond 65535 (e.g. up to 128k).
# Writer and reader share this so the two never diverge.
DTYPE = np.uint32

# Default shard size: 2**28 uint32 tokens = 1 GiB per shard file.
DEFAULT_SHARD_TOKENS = 256 * 2**20


class ShardWriter:
    """Splits a continuous token stream across fixed-size shard files.

    Writes 00000.bin, 00001.bin, ... under `out_dir`, each holding at most
    `shard_tokens` tokens. Any *.bin already in `out_dir` is deleted first --
    the sharded equivalent of truncating a single output file with mode "wb";
    shards left over from a previous, longer run would otherwise still be
    picked up as valid data by PackedDataset.
    """

    def __init__(self, out_dir: str | Path, shard_tokens: int = DEFAULT_SHARD_TOKENS):
        assert shard_tokens > 0
        self.out_dir = Path(out_dir)
        self.shard_tokens = shard_tokens
        self.n_shards = 0
        self._in_shard = 0
        self._file = None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.out_dir.glob("*.bin"):
            stale.unlink()

    def write(self, tokens: np.ndarray) -> None:
        tokens = np.asarray(tokens, dtype=DTYPE)
        while tokens.size:
            if self._file is None:  # shards are created lazily: none are empty
                self._file = open(self.out_dir / f"{self.n_shards:05d}.bin", "wb")
                self.n_shards += 1
                self._in_shard = 0
            room = self.shard_tokens - self._in_shard
            head, tokens = tokens[:room], tokens[room:]
            head.tofile(self._file)
            self._in_shard += head.size
            if self._in_shard >= self.shard_tokens:
                self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class PackedDataset(Dataset):
    def __init__(self, path: str, block_size: int = 1024, random: bool = True):
        """
        Args:
            path: token binary produced by base_setup.py -- a directory of
                *.bin shards. A single .bin file (or a bare `foo` resolving to
                a legacy `foo.bin`) is also accepted for pre-sharding corpora.
            block_size: effective context length. Each sample is block_size+1 tokens.
            random: True for random offsets, False for non-overlapping contiguous
                blocks.
        """
        self.block_size = block_size
        self.random = random
        p = Path(path)
        if p.is_dir():
            shard_paths = sorted(p.glob("*.bin"))
            if not shard_paths:
                raise FileNotFoundError(f"no .bin shards in directory {path}")
        elif p.is_file():
            shard_paths = [p]
        elif Path(str(p) + ".bin").is_file():  # legacy single-file layout
            shard_paths = [Path(str(p) + ".bin")]
        else:
            raise FileNotFoundError(f"no token binary at {path}")
        self.paths = [str(q) for q in shard_paths]
        itemsize = np.dtype(DTYPE).itemsize
        shard_tokens = [q.stat().st_size // itemsize for q in shard_paths]
        self.n_tokens = int(sum(shard_tokens))
        # Samples never cross a shard boundary: shards are separate files, and
        # at ~GiB size the block_size-1 windows lost per boundary are
        # negligible. A shard shorter than one sample contributes nothing.
        if random:
            counts = [max(0, n - block_size) for n in shard_tokens]
        else:
            counts = [n // (block_size + 1) for n in shard_tokens]
        self._cum_counts = np.cumsum(counts)
        # Memmaps are opened lazily per DataLoader worker process: one opened
        # in __init__ would share its file descriptor across the worker fork.
        self._mmaps: list[np.memmap | None] = [None] * len(self.paths)
        assert len(self) > 0, (
            f"corpus at {path} ({self.n_tokens} tokens) has no shard with a "
            f"full block_size+1 ({block_size + 1}) sample"
        )

    def _shard(self, i: int) -> np.memmap:
        if self._mmaps[i] is None:
            self._mmaps[i] = np.memmap(self.paths[i], dtype=DTYPE, mode="r")
        return self._mmaps[i]

    def __len__(self) -> int:
        return int(self._cum_counts[-1])

    def __getitem__(self, idx: int) -> Tensor:
        shard = int(np.searchsorted(self._cum_counts, idx, side="right"))
        local = int(idx - (self._cum_counts[shard - 1] if shard else 0))
        start = local if self.random else local * (self.block_size + 1)
        chunk = self._shard(shard)[start : start + self.block_size + 1]
        return torch.from_numpy(chunk.astype(np.int64))


# Number of indices a chunked sampler draws per iteration. Bounds peak memory:
# both samplers below are asked for num_samples == len(train_ds) == the whole
# token count (billions for a large corpus). Producing all of them at once --
# num_samples-sized tensors plus a num_samples-long Python list from .tolist()
# -- costs tens of GB before the first batch is even yielded, which is what
# exhausts host memory and gets the process OOM-killed. Drawing in fixed-size
# chunks keeps peak memory O(_SAMPLE_CHUNK) regardless of corpus size; training
# stops at max_steps long before an "epoch" of num_samples is consumed.
_SAMPLE_CHUNK = 1 << 20


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
        remaining = self.num_samples
        while remaining > 0:
            n = min(_SAMPLE_CHUNK, remaining)
            group_ids = torch.multinomial(self.group_weights, n, replacement=True)
            local = (torch.rand(n, dtype=torch.double) * self.group_sizes[group_ids]).long()
            idx = self.group_offsets[group_ids] + local
            yield from idx.tolist()
            remaining -= n


class UniformIndexSampler(Sampler[int]):
    """Draws indices uniformly at random with replacement, in fixed-size chunks.

    Used for the unweighted train path in place of DataLoader(shuffle=True),
    whose RandomSampler materializes a full `torch.randperm(len)` (then
    `.tolist()`) up front. `len` is the whole token count for a large corpus,
    so that eager permutation exhausts host memory exactly like the weighted
    sampler did (see _SAMPLE_CHUNK / GroupWeightedIndexSampler). Sampling with
    replacement is fine here: examples are random-offset windows into a token
    stream, and training stops at max_steps well before num_samples is drawn.
    """

    def __init__(self, dataset_len: int, num_samples: int):
        self.dataset_len = dataset_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        remaining = self.num_samples
        while remaining > 0:
            n = min(_SAMPLE_CHUNK, remaining)
            idx = torch.randint(self.dataset_len, (n,))
            yield from idx.tolist()
            remaining -= n


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
        # Not shuffle=True: DataLoader's RandomSampler would build a full
        # torch.randperm(len(train_ds)) up front, which OOMs on a large corpus
        # for the same reason the weighted path did. UniformIndexSampler draws
        # the same uniform indices lazily in bounded-size chunks.
        sampler = UniformIndexSampler(len(self.train_ds), num_samples=len(self.train_ds))
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
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
