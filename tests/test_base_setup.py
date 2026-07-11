import pytest

from scripts.base_setup import expand_val_fraction


def test_expand_val_fraction_passes_through_plain_entries():
    entries = [{"preset": "tinystories", "output": "tinystories"}]
    assert expand_val_fraction(entries) == entries


def test_expand_val_fraction_splits_into_train_and_val_entries():
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
        {"preset": "wikipedia-en", "output": "wikipedia-en", "split": "train[1%:]"},
        {"preset": "wikipedia-en", "output": "wikipedia-en.val", "split": "train[:1%]"},
    ]


def test_expand_val_fraction_respects_explicit_base_split():
    entries = [
        {
            "preset": "cosmopedia",
            "split": "web_samples_v2",
            "output": "cosmopedia",
            "val_output": "cosmopedia.val",
            "val_fraction": 0.002,
        }
    ]
    expanded = expand_val_fraction(entries)
    assert expanded[0]["split"] == "web_samples_v2[0.2%:]"
    assert expanded[1]["split"] == "web_samples_v2[:0.2%]"


def test_expand_val_fraction_requires_val_output():
    entries = [
        {"preset": "wikipedia-en", "output": "wikipedia-en", "val_fraction": 0.01}
    ]
    with pytest.raises(SystemExit):
        expand_val_fraction(entries)
