"""scripts/preprocess.py が生成した flat uint16 バイナリを読む Dataset。

ファイルは padding なしで連結された連続トークン列。block_size+1 の窓でスライスして
返すと、GPT._loss が内部で1つシフトして次トークン予測の損失を計算する
（系列長 block_size+1 -> 実効コンテキスト block_size）。
"""

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

DTYPE = np.uint16


class PackedDataset(Dataset):
    def __init__(self, path: str, block_size: int = 1024, random: bool = True):
        """
        Args:
            path: preprocess.py が出力した .bin ファイル
            block_size: 実効コンテキスト長。各サンプルは block_size+1 トークン。
            random: True ならランダムオフセット、False なら非重複の連続ブロック。
        """
        self.path = path
        self.block_size = block_size
        self.random = random
        # 長さだけ先に確定させる。memmap 本体は worker fork 後に開く（下記）。
        n = np.memmap(path, dtype=DTYPE, mode="r").shape[0]
        self.n_tokens = int(n)
        self._data: np.memmap | None = None
        assert self.n_tokens > block_size, (
            f"corpus ({self.n_tokens} tokens) が block_size+1 ({block_size + 1}) より短い"
        )

    @property
    def data(self) -> np.memmap:
        # DataLoader の worker プロセスごとに開き直す（memmap を __init__ で持つと
        # fork 後にファイルディスクリプタを共有して壊れることがある）。
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
