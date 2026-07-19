"""The training CLIs' config-to-dataset wiring (scripts/base_train.py,
scripts/sft_train.py) and the checkpoint-loading helpers in picochat.trainer
they build on."""

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset

from picochat.dataloader import DTYPE, ShardWriter, write_meta
from picochat.presets import build_lm
from picochat.trainer import GPT, load_lm_from_checkpoint
from scripts import base_train, sft_train

BLOCK_SIZE = 8


def _shard_dir(tmp_path, name: str, n_rows: int = 4):
    """A minimal PackedDataset corpus: n_rows rows of BLOCK_SIZE+1 tokens."""
    out = tmp_path / name
    writer = ShardWriter(out)
    writer.write(np.arange(n_rows * (BLOCK_SIZE + 1), dtype=DTYPE))
    writer.close()
    write_meta(out, BLOCK_SIZE)
    return str(out)


def _sft_bundle(tmp_path, name: str, n_rows: int = 4):
    """A minimal SFTTensorDataset bundle, as written by scripts/sft_setup.py."""
    path = tmp_path / name
    rows = torch.zeros(n_rows, BLOCK_SIZE, dtype=torch.long)
    torch.save(
        {
            "input_ids": rows,
            "labels": rows.clone(),
            "doc_ids": rows.clone(),
            "pad_id": 0,
        },
        path,
    )
    return str(path)


# ---------------------------------------------------------------------------
# base_train: config path resolution and dataset assembly
# ---------------------------------------------------------------------------
def test_resolve_bins_joins_data_dir():
    assert base_train.resolve_bins("a", "data") == ["data/a"]
    assert base_train.resolve_bins(["a", "b"], "data") == ["data/a", "data/b"]


def test_resolve_datasets_splits_paths_and_weights():
    paths, weights = base_train.resolve_datasets(
        [{"path": "x", "weight": 2.0}, {"path": "y"}], "data"
    )
    assert paths == ["data/x", "data/y"]
    assert weights == [2.0, 1.0]  # weight defaults to 1.0


def test_make_dataset_single_and_weighted(tmp_path):
    d1 = _shard_dir(tmp_path, "one", n_rows=3)
    d2 = _shard_dir(tmp_path, "two", n_rows=5)

    ds, weights = base_train.make_dataset(d1, BLOCK_SIZE)
    assert len(ds) == 3 and weights is None

    ds, weights = base_train.make_dataset([d1, d2], BLOCK_SIZE, weights=[0.7, 0.3])
    assert isinstance(ds, ConcatDataset) and len(ds) == 8
    assert weights == [0.7, 0.3]


def test_make_dataset_rejects_bad_weights(tmp_path):
    d1 = _shard_dir(tmp_path, "one")
    d2 = _shard_dir(tmp_path, "two")
    with pytest.raises(ValueError):  # weights need more than one source
        base_train.make_dataset(d1, BLOCK_SIZE, weights=[1.0])
    with pytest.raises(ValueError):  # one weight per source
        base_train.make_dataset([d1, d2], BLOCK_SIZE, weights=[1.0])


# ---------------------------------------------------------------------------
# sft_train: same wiring over .pt bundles
# ---------------------------------------------------------------------------
def test_sft_resolve_paths_and_make_dataset(tmp_path):
    p1 = _sft_bundle(tmp_path, "a.pt", n_rows=2)
    p2 = _sft_bundle(tmp_path, "b.pt", n_rows=3)

    paths, weights = sft_train.resolve_paths(
        [{"path": "a.pt", "weight": 0.5}, {"path": "b.pt"}], str(tmp_path)
    )
    assert paths == [p1, p2] and weights == [0.5, 1.0]

    ds, group_weights = sft_train.make_dataset([p1, p2], weights=[0.5, 1.0])
    assert isinstance(ds, ConcatDataset) and len(ds) == 5
    assert group_weights == [0.5, 1.0]

    with pytest.raises(ValueError):
        sft_train.make_dataset([p1], weights=[1.0])
    with pytest.raises(ValueError):
        sft_train.make_dataset([p1, p2], weights=[1.0])


# ---------------------------------------------------------------------------
# checkpoint round-trip: the model_config saved by GPT rebuilds the same model
# ---------------------------------------------------------------------------
def _tiny_model_config():
    return dict(
        size="1b",
        vocab_size=64,
        max_seq_len=32,
        d_model=16,
        n_heads=2,
        n_kv_heads=2,
        n_layers=1,
    )


def _save_ckpt(tmp_path, model_config):
    lm = build_lm(**model_config)
    gpt = GPT(lm, compile=False, model_config=model_config)
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "state_dict": gpt.state_dict(),
            "hyper_parameters": {"model_config": model_config},
        },
        path,
    )
    return str(path), lm


def test_load_lm_from_checkpoint_round_trip(tmp_path):
    model_config = _tiny_model_config()
    path, lm = _save_ckpt(tmp_path, model_config)
    loaded, loaded_config = load_lm_from_checkpoint(path, vocab_size=64)
    assert loaded_config == model_config
    for k, v in lm.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v), k


def test_load_lm_from_checkpoint_applies_overrides(tmp_path):
    path, _ = _save_ckpt(tmp_path, _tiny_model_config())
    # continual learning: max_seq_len raised at load time (RoPE tables are
    # rebuilt from it; no learned tensor changes shape)
    loaded, cfg = load_lm_from_checkpoint(
        path, vocab_size=64, overrides={"max_seq_len": 64}
    )
    assert cfg["max_seq_len"] == 64


def test_load_lm_from_checkpoint_rejects_bad_files(tmp_path):
    not_lightning = tmp_path / "plain.pt"
    torch.save({"weights": 1}, not_lightning)
    with pytest.raises(ValueError, match="Lightning checkpoint"):
        load_lm_from_checkpoint(str(not_lightning), vocab_size=64)

    no_config = tmp_path / "no_config.pt"
    torch.save({"state_dict": {}, "hyper_parameters": {}}, no_config)
    with pytest.raises(ValueError, match="model_config"):
        load_lm_from_checkpoint(str(no_config), vocab_size=64)
