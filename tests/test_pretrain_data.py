import lightning as L
import pytest
import torch
from lightning.pytorch.utilities.model_helpers import is_overridden
from torch.utils.data import Dataset

from picochat.data.pretrain import PretrainDataModule, holdout_splits


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
    from torch.utils.data import ConcatDataset

    ds = ConcatDataset([small, large])
    dm = PretrainDataModule(
        ds, None, batch_size=64, num_workers=0, train_group_weights=[1.0, 1.0]
    )

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


def test_train_dataloader_unweighted_uses_chunked_uniform_sampler():
    # Without weights the loader must not fall back to DataLoader(shuffle=True):
    # its RandomSampler materializes a full randperm(len) up front, which OOMs
    # on a large corpus. It should use the lazy, in-range UniformIndexSampler.
    from picochat.data.pretrain import UniformIndexSampler

    ds = _RandomTokenDataset(40, 6, n=32)
    dm = PretrainDataModule(ds, None, batch_size=4, num_workers=0)
    loader = dm.train_dataloader()
    assert isinstance(loader.sampler, UniformIndexSampler)
    drawn = list(iter(loader.sampler))
    assert len(drawn) == len(ds)
    assert all(0 <= i < len(ds) for i in drawn)


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


# ---------------------------------------------------------------------------
# ShardWriter / sharded PackedDataset
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

from picochat.data.pretrain import DTYPE, PackedDataset, ShardWriter  # noqa: E402


def _read_all(shard_dir):
    files = sorted(shard_dir.glob("*.bin"))
    return files, np.concatenate([np.fromfile(f, dtype=DTYPE) for f in files])


def test_shard_writer_splits_at_shard_tokens(tmp_path):
    w = ShardWriter(tmp_path / "ds", shard_tokens=10)
    w.write(np.arange(25, dtype=DTYPE))
    w.close()
    files, got = _read_all(tmp_path / "ds")
    assert [f.name for f in files] == ["00000.bin", "00001.bin", "00002.bin"]
    itemsize = np.dtype(DTYPE).itemsize
    assert [f.stat().st_size // itemsize for f in files] == [10, 10, 5]
    # concatenating the shards reproduces the original stream exactly
    assert (got == np.arange(25, dtype=DTYPE)).all()


def test_shard_writer_write_chunks_smaller_than_shard(tmp_path):
    # several write() calls, none aligned with the shard boundary
    w = ShardWriter(tmp_path / "ds", shard_tokens=8)
    stream = np.arange(20, dtype=DTYPE)
    for chunk in np.split(stream, [3, 9, 15]):
        w.write(chunk)
    w.close()
    files, got = _read_all(tmp_path / "ds")
    itemsize = np.dtype(DTYPE).itemsize
    assert [f.stat().st_size // itemsize for f in files] == [8, 8, 4]
    assert (got == stream).all()


def test_shard_writer_removes_stale_shards(tmp_path):
    # a rerun producing fewer shards must not leave old shards behind, or the
    # reader would silently mix stale data into the corpus
    d = tmp_path / "ds"
    w = ShardWriter(d, shard_tokens=4)
    w.write(np.arange(12, dtype=DTYPE))  # 3 shards
    w.close()
    w = ShardWriter(d, shard_tokens=4)
    w.write(np.arange(4, dtype=DTYPE))  # 1 shard
    w.close()
    assert [f.name for f in sorted(d.glob("*.bin"))] == ["00000.bin"]


def test_packed_dataset_reads_shard_directory_contiguous(tmp_path):
    w = ShardWriter(tmp_path / "ds", shard_tokens=10)
    w.write(np.arange(30, dtype=DTYPE))
    w.close()
    ds = PackedDataset(str(tmp_path / "ds"), block_size=4, random=False)
    # 10 tokens per shard -> two non-overlapping 5-token blocks per shard
    assert len(ds) == 6
    assert ds[0].tolist() == [0, 1, 2, 3, 4]
    assert ds[2].tolist() == [10, 11, 12, 13, 14]  # first block of shard 1
    assert ds[5].tolist() == [25, 26, 27, 28, 29]  # last block of shard 2


def test_packed_dataset_random_windows_stay_within_shard(tmp_path):
    w = ShardWriter(tmp_path / "ds", shard_tokens=10)
    w.write(np.arange(30, dtype=DTYPE))
    w.close()
    ds = PackedDataset(str(tmp_path / "ds"), block_size=4, random=True)
    # 10 - 4 = 6 random windows per shard
    assert len(ds) == 18
    for i in range(len(ds)):
        chunk = ds[i]
        # tokens are arange, so a valid window is 5 consecutive values...
        assert (chunk[1:] - chunk[:-1] == 1).all()
        # ...that never cross a shard boundary (each shard is one decade)
        assert (chunk // 10 == chunk[0] // 10).all()


def test_packed_dataset_skips_shard_shorter_than_one_sample(tmp_path):
    # 13 tokens with shard_tokens=10 -> final shard has 3 tokens < block+1
    w = ShardWriter(tmp_path / "ds", shard_tokens=10)
    w.write(np.arange(13, dtype=DTYPE))
    w.close()
    ds = PackedDataset(str(tmp_path / "ds"), block_size=4, random=True)
    assert ds.n_tokens == 13
    assert len(ds) == 6  # only shard 0 contributes samples
    for i in range(len(ds)):
        assert ds[i].max() < 10


def test_packed_dataset_single_file_still_works(tmp_path):
    f = tmp_path / "corpus.bin"
    np.arange(20, dtype=DTYPE).tofile(f)
    ds = PackedDataset(str(f), block_size=4, random=False)
    assert len(ds) == 4
    assert ds[0].tolist() == [0, 1, 2, 3, 4]


def test_packed_dataset_legacy_bin_fallback(tmp_path):
    # config says `corpus` but only the pre-sharding `corpus.bin` exists
    np.arange(20, dtype=DTYPE).tofile(tmp_path / "corpus.bin")
    ds = PackedDataset(str(tmp_path / "corpus"), block_size=4)
    assert ds.n_tokens == 20


def test_packed_dataset_missing_path_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        PackedDataset(str(tmp_path / "nope"), block_size=4)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        PackedDataset(str(tmp_path / "empty"), block_size=4)


# ---------------------------------------------------------------------------
# holdout_splits: carving a validation slice out of a train-only split
# ---------------------------------------------------------------------------


def test_holdout_splits_partitions_by_absolute_count():
    # absolute-index slicing, not percentage: this project's pinned `datasets`
    # version only parses whole-number percentages (see holdout_splits'
    # docstring), too coarse for the sub-1% fractions used on huge datasets
    train_split, val_split = holdout_splits("train", 0.01, total_examples=1000)
    assert train_split == "train[10:]"
    assert val_split == "train[:10]"


def test_holdout_splits_rounds_up_to_at_least_one_example():
    train_split, val_split = holdout_splits("train", 0.001, total_examples=100)
    assert val_split == "train[:1]"
    assert train_split == "train[1:]"


def test_holdout_splits_preserves_base_split_name():
    train_split, val_split = holdout_splits(
        "web_samples_v2", 0.002, total_examples=10000
    )
    assert train_split.startswith("web_samples_v2[")
    assert val_split.startswith("web_samples_v2[:")


def test_holdout_splits_rejects_out_of_range_fraction():
    with pytest.raises(AssertionError):
        holdout_splits("train", 0.0, total_examples=100)
    with pytest.raises(AssertionError):
        holdout_splits("train", 1.0, total_examples=100)


def test_holdout_splits_produces_split_strings_datasets_can_parse():
    # regression guard for the exact bug this was built to catch: percentage
    # slicing silently producing a string (e.g. "train[0.1%:]") that looks
    # valid but datasets.arrow_reader rejects at load_dataset() time
    from datasets.arrow_reader import _SUB_SPEC_RE

    train_split, val_split = holdout_splits("train", 0.0005, total_examples=6_407_814)
    assert _SUB_SPEC_RE.match(train_split)
    assert _SUB_SPEC_RE.match(val_split)
