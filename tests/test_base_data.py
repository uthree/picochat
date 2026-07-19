import lightning as L
import pytest
import torch
from lightning.pytorch.utilities.model_helpers import is_overridden
from torch.utils.data import Dataset

from picochat.dataloader import PretrainDataModule
from picochat.dataset import (
    CHAT_PRESETS,
    TEXT_PRESETS,
    DatasetSpec,
    Mixture,
    chat_spec_from_entry,
    holdout_splits,
    iter_mixture,
    iter_texts,
    resolve_text_spec,
    spec_from_entry,
)


def test_spec_from_entry_preset_inline_and_split_override():
    # preset lookup
    preset_name = next(iter(TEXT_PRESETS))
    assert spec_from_entry({"preset": preset_name}) is TEXT_PRESETS[preset_name]
    # inline spec
    inline = spec_from_entry({"path": "x", "split": "test", "text_key": "body"})
    assert inline.path == "x" and inline.split == "test" and inline.text_key == "body"
    # per-entry split override copies rather than mutating the shared preset
    original_split = TEXT_PRESETS[preset_name].split
    overridden = spec_from_entry({"preset": preset_name, "split": "validation"})
    assert overridden.split == "validation"
    assert TEXT_PRESETS[preset_name].split == original_split  # preset untouched
    # bad entry
    with pytest.raises(SystemExit):
        spec_from_entry({})


def test_chat_spec_from_entry_inline_and_error():
    spec = chat_spec_from_entry({"path": "y", "messages_key": "conv"})
    assert spec.path == "y" and spec.messages_key == "conv"
    with pytest.raises(SystemExit):
        chat_spec_from_entry({"preset": "does-not-exist"})


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
    from picochat.dataloader import UniformIndexSampler

    ds = _RandomTokenDataset(40, 6, n=32)
    dm = PretrainDataModule(ds, None, batch_size=4, num_workers=0)
    loader = dm.train_dataloader()
    assert isinstance(loader.sampler, UniformIndexSampler)
    drawn = list(iter(loader.sampler))
    assert len(drawn) == len(ds)
    assert all(0 <= i < len(ds) for i in drawn)


def test_samplers_seeded_streams_are_deterministic_and_rank_distinct():
    # Per-rank seeding is how multi-GPU decorrelates the IID samplers (see
    # PretrainDataModule): same seed -> same stream, different seed (rank) ->
    # a different stream.
    from picochat.dataloader import GroupWeightedIndexSampler, UniformIndexSampler

    def uniform(seed):
        return list(iter(UniformIndexSampler(1000, 64, seed=seed)))

    assert uniform(7) == uniform(7)
    assert uniform(7) != uniform(8)

    def weighted(seed):
        return list(
            iter(GroupWeightedIndexSampler([10, 990], [1.0, 1.0], 64, seed=seed))
        )

    assert weighted(7) == weighted(7)
    assert weighted(7) != weighted(8)


def test_sampler_reiteration_continues_the_stream():
    # A second "epoch" must not replay the first: the generator persists across
    # __iter__ calls (a replayed stream would train on the same batches twice).
    from picochat.dataloader import UniformIndexSampler

    sampler = UniformIndexSampler(1000, 32, seed=7)
    assert list(iter(sampler)) != list(iter(sampler))


def test_datamodule_passes_per_rank_seed(monkeypatch):
    # Outside distributed runs the sampler gets the base seed; under a
    # (mocked) 2-rank process group, rank 1 draws from seed + 1.
    ds = _RandomTokenDataset(40, 6, n=32)
    dm = PretrainDataModule(ds, None, batch_size=4, num_workers=0, seed=7)
    assert dm.train_dataloader().sampler.seed == 7

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    assert dm.train_dataloader().sampler.seed == 8


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
# ShardWriter / pack_docs / sharded PackedDataset
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

from picochat.dataloader import (  # noqa: E402
    DTYPE,
    PackedDataset,
    ShardWriter,
    pack_docs,
    write_meta,
)

BOS, EOS, PAD = 1, 2, 0


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


def _doc(*body):
    return [BOS, *body, EOS]


def test_pack_docs_packs_several_docs_per_row():
    # block_size=7 -> rows of 8 tokens; a 4-token and a 3-token doc share one
    rows = pack_docs([_doc(10, 11), _doc(12)], block_size=7, pad_id=PAD, bos_id=BOS)
    assert rows.shape == (1, 8)
    assert rows[0].tolist() == [BOS, 10, 11, EOS, BOS, 12, EOS, PAD]


def test_pack_docs_pads_leftover_room():
    rows = pack_docs([_doc(10, 11)], block_size=7, pad_id=PAD, bos_id=BOS)
    assert rows[0].tolist() == [BOS, 10, 11, EOS, PAD, PAD, PAD, PAD]


def test_pack_docs_splits_long_doc_into_bos_prefixed_chunks():
    # 10-token doc into rows of 5: chunk 0 takes 5 tokens, continuations take
    # 4 stream tokens behind a fresh BOS so the doc-ids-from-BOS derivation
    # sees each chunk as its own document
    doc = _doc(*range(10, 18))  # BOS 10..17 EOS
    rows = pack_docs([doc], block_size=4, pad_id=PAD, bos_id=BOS)
    chunks = sorted(rows.tolist())
    assert [BOS, 10, 11, 12, 13] in chunks
    assert [BOS, 14, 15, 16, 17] in chunks
    assert [BOS, EOS, PAD, PAD, PAD] in chunks
    # every chunk starts with BOS and no stream token was lost or duplicated
    flat = [t for row in rows.tolist() for t in row if t != PAD and t != BOS]
    assert sorted(flat) == sorted([*range(10, 18), EOS])


def test_pack_docs_keeps_every_document():
    docs = [_doc(*range(10, 10 + n)) for n in (1, 2, 3, 5, 8, 13)]
    rows = pack_docs(docs, block_size=15, pad_id=PAD, bos_id=BOS)
    flat = [t for row in rows.tolist() for t in row]
    total = sum(len(d) for d in docs)
    assert sum(1 for t in flat if t != PAD) == total
    assert sum(1 for t in flat if t == BOS) == len(docs)  # nothing was split


def _write_corpus(shard_dir, docs, block_size, shard_tokens=None):
    rows = pack_docs(docs, block_size, pad_id=PAD, bos_id=BOS)
    w = ShardWriter(shard_dir, shard_tokens or rows.size)
    w.write(rows.reshape(-1))
    w.close()
    write_meta(shard_dir, block_size)
    return rows


def test_packed_dataset_returns_one_row_per_item(tmp_path):
    docs = [_doc(*range(10, 10 + n)) for n in (2, 3, 4, 5)]
    rows = _write_corpus(tmp_path / "ds", docs, block_size=7)
    ds = PackedDataset(str(tmp_path / "ds"), block_size=7)
    assert len(ds) == len(rows)
    got = sorted(ds[i].tolist() for i in range(len(ds)))
    assert got == sorted(rows.tolist())


def test_packed_dataset_rows_never_straddle_shards(tmp_path):
    # shard_tokens equal to one row -> one row per shard file, read back intact
    docs = [_doc(10, 11, 12), _doc(13, 14, 15), _doc(16, 17, 18)]
    rows = _write_corpus(tmp_path / "ds", docs, block_size=5, shard_tokens=6)
    assert len(list((tmp_path / "ds").glob("*.bin"))) == len(rows)
    ds = PackedDataset(str(tmp_path / "ds"), block_size=5)
    got = sorted(ds[i].tolist() for i in range(len(ds)))
    assert got == sorted(rows.tolist())


def test_packed_dataset_rejects_mismatched_block_size(tmp_path):
    _write_corpus(tmp_path / "ds", [_doc(10, 11)], block_size=7)
    with pytest.raises(ValueError, match="block_size"):
        PackedDataset(str(tmp_path / "ds"), block_size=4)


def test_packed_dataset_requires_meta(tmp_path):
    d = tmp_path / "ds"
    w = ShardWriter(d, shard_tokens=8)
    w.write(np.arange(8, dtype=DTYPE))
    w.close()
    with pytest.raises(FileNotFoundError, match="meta.json"):
        PackedDataset(str(d), block_size=7)


def test_packed_dataset_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PackedDataset(str(tmp_path / "nope"), block_size=4)
    (tmp_path / "empty").mkdir()
    write_meta(tmp_path / "empty", 4)
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


# ---------------------------------------------------------------------------
# resolve_text_spec (CLI --preset/--dataset resolution)
# ---------------------------------------------------------------------------
def test_resolve_text_spec_preset_returns_a_copy():
    name = next(iter(TEXT_PRESETS))
    spec = resolve_text_spec(name, None)
    assert spec == TEXT_PRESETS[name]
    assert spec is not TEXT_PRESETS[name]
    # base_setup.py's --split override mutates the returned spec; the shared
    # preset must stay untouched
    original = TEXT_PRESETS[name].split
    spec.split = "mutated"
    assert TEXT_PRESETS[name].split == original


def test_resolve_text_spec_inline_and_defaults():
    spec = resolve_text_spec(None, "repo:cfg:val:body")
    assert (spec.path, spec.name, spec.split, spec.text_key) == (
        "repo",
        "cfg",
        "val",
        "body",
    )
    # omitted segments fall back to defaults
    spec = resolve_text_spec(None, "repo")
    assert (spec.name, spec.split, spec.text_key) == (None, "train", "text")


def test_resolve_text_spec_errors():
    with pytest.raises(SystemExit):
        resolve_text_spec("no-such-preset", None)
    with pytest.raises(SystemExit):
        resolve_text_spec(None, None)


def test_chat_spec_from_entry_split_override_copies():
    name = next(iter(CHAT_PRESETS))
    original = CHAT_PRESETS[name].split
    spec = chat_spec_from_entry({"preset": name, "split": "held-out"})
    assert spec.split == "held-out"
    assert CHAT_PRESETS[name].split == original  # preset untouched
    with pytest.raises(SystemExit):
        chat_spec_from_entry({})


# ---------------------------------------------------------------------------
# iter_texts / iter_mixture character budgeting (drives tokenizer byte
# balancing); load_dataset is faked so no network is touched
# ---------------------------------------------------------------------------
def test_iter_texts_max_chars_includes_budget_crossing_doc(monkeypatch):
    texts = ["aaaa", "bbbb", "cccc"]
    monkeypatch.setattr(
        "picochat.dataset.load_dataset",
        lambda *a, **k: [{"text": t} for t in texts],
    )
    spec = DatasetSpec("fake")
    # yield-then-check: the doc that crosses the budget is still yielded
    assert list(iter_texts(spec, max_chars=5)) == ["aaaa", "bbbb"]
    # no budget -> everything (empty/whitespace-only docs are skipped)
    assert list(iter_texts(spec)) == texts


def test_iter_texts_skips_empty_docs(monkeypatch):
    monkeypatch.setattr(
        "picochat.dataset.load_dataset",
        lambda *a, **k: [{"text": "x"}, {"text": ""}, {"text": "  "}, {"text": "y"}],
    )
    assert list(iter_texts(DatasetSpec("fake"))) == ["x", "y"]


def test_iter_mixture_budgets_chars_per_source(monkeypatch):
    monkeypatch.setattr(
        "picochat.dataset.load_dataset",
        lambda *a, **k: [{"text": "x" * 10} for _ in range(100)],
    )
    mix = Mixture(
        specs=[DatasetSpec("a"), DatasetSpec("b")],
        weights=[0.75, 0.25],
    )
    got = list(iter_mixture(mix, total_chars=80))
    # source a reads 0.75*80=60 chars -> 6 docs; source b 20 chars -> 2 docs
    assert len(got) == 8
