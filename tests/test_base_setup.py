import pytest

from scripts import base_setup
from scripts.base_setup import expand_val_fraction


class _FakeSplitInfo:
    def __init__(self, num_examples: int):
        self.num_examples = num_examples


class _FakeInfo:
    def __init__(self, splits: dict[str, int]):
        self.splits = {name: _FakeSplitInfo(n) for name, n in splits.items()}


class _FakeBuilder:
    def __init__(self, splits: dict[str, int]):
        self.info = _FakeInfo(splits)


def _stub_builder(splits: dict[str, int], monkeypatch):
    """expand_val_fraction resolves total_examples via load_dataset_builder
    (a Hub metadata call) -- stub it so these stay offline unit tests."""
    monkeypatch.setattr(
        base_setup, "load_dataset_builder", lambda path, name: _FakeBuilder(splits)
    )


def test_expand_val_fraction_passes_through_plain_entries():
    entries = [{"preset": "tinystories", "output": "tinystories"}]
    assert expand_val_fraction(entries) == entries


def test_expand_val_fraction_splits_into_train_and_val_entries(monkeypatch):
    _stub_builder({"train": 1000}, monkeypatch)
    entries = [
        {
            "preset": "wikipedia-en",
            "output": "wikipedia-en",
            "val_output": "wikipedia-en.val",
            "val_fraction": 0.01,
        }
    ]
    expanded = expand_val_fraction(entries)
    assert expanded == [
        {"preset": "wikipedia-en", "output": "wikipedia-en", "split": "train[10:]"},
        {"preset": "wikipedia-en", "output": "wikipedia-en.val", "split": "train[:10]"},
    ]


def test_expand_val_fraction_respects_explicit_base_split(monkeypatch):
    _stub_builder({"test": 500}, monkeypatch)
    entries = [
        {
            "preset": "cosmopedia",
            "split": "test",
            "output": "cosmopedia",
            "val_output": "cosmopedia.val",
            "val_fraction": 0.002,
        }
    ]
    expanded = expand_val_fraction(entries)
    assert expanded[0]["split"] == "test[1:]"
    assert expanded[1]["split"] == "test[:1]"


def test_expand_val_fraction_requires_val_output():
    entries = [
        {"preset": "wikipedia-en", "output": "wikipedia-en", "val_fraction": 0.01}
    ]
    with pytest.raises(SystemExit):
        expand_val_fraction(entries)


def test_expand_val_fraction_reports_a_clear_error_when_count_lookup_fails(
    monkeypatch,
):
    def _boom(path, name):
        raise ValueError("nope")

    monkeypatch.setattr(base_setup, "load_dataset_builder", _boom)
    entries = [
        {
            "preset": "wikipedia-en",
            "output": "wikipedia-en",
            "val_output": "wikipedia-en.val",
            "val_fraction": 0.01,
        }
    ]
    with pytest.raises(SystemExit):
        expand_val_fraction(entries)
