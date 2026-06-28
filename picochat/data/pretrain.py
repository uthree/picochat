"""Dataset that reads the flat token binary produced by scripts/preprocess.py.

The file is a continuous token stream concatenated without padding. Slicing a
block_size+1 window and returning it lets GPT._loss shift by one internally to
compute the next-token prediction loss (sequence length block_size+1 ->
effective context block_size).
"""

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# 32-bit token ids: leaves headroom for vocab beyond 65535 (e.g. up to 128k).
# Writer (scripts/preprocess.py) imports this so the two never diverge.
DTYPE = np.uint32


class PackedDataset(Dataset):
    def __init__(self, path: str, block_size: int = 1024, random: bool = True):
        """
        Args:
            path: the .bin file produced by preprocess.py
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
