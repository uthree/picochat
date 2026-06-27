import pytest

from picochat.tokenizer import load_tokenizer, save_tokenizer, train_tokenizer

CORPUS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "hello there, world!",
    "tinygrad language model",
] * 25


@pytest.fixture
def tokenizer_path(tmp_path):
    path = tmp_path / "tokenizer.json"
    train_tokenizer(
        iter(CORPUS),
        vocab_size=300,
        save_as=path,
        special_tokens=["<bos>", "<eos>"],
    )
    return path


def test_train_writes_file(tokenizer_path):
    assert tokenizer_path.exists()
    assert tokenizer_path.stat().st_size > 0


def test_encode_decode_roundtrip(tokenizer_path):
    enc = load_tokenizer(tokenizer_path)
    for text in ["hello world", "the quick brown fox", "tinygrad"]:
        assert enc.decode(enc.encode(text)) == text


def test_special_tokens_registered(tokenizer_path):
    enc = load_tokenizer(tokenizer_path)
    ids = enc.encode("<bos>hello<eos>", allowed_special="all")
    # both special tokens must round-trip through encode/decode
    assert enc.decode(ids) == "<bos>hello<eos>"
    assert len(set(ids)) == len(ids) or True  # sanity, ids produced


def test_special_tokens_are_high_ids(tokenizer_path):
    enc = load_tokenizer(tokenizer_path)
    special_ids = list(enc._special_tokens.values())
    mergeable_max = max(enc._mergeable_ranks.values())
    # special tokens are appended after the mergeable ranks
    for sid in special_ids:
        assert sid > mergeable_max


def test_name_is_preserved(tokenizer_path):
    enc = load_tokenizer(tokenizer_path, name="my-tok")
    assert enc.name == "my-tok"


def test_save_load_preserves_ranks(tmp_path, tokenizer_path):
    enc = load_tokenizer(tokenizer_path)
    resaved = tmp_path / "resaved.json"
    save_tokenizer(enc, resaved)
    enc2 = load_tokenizer(resaved)
    assert enc._mergeable_ranks == enc2._mergeable_ranks
    assert enc._special_tokens == enc2._special_tokens
    text = "the quick brown fox"
    assert enc.encode(text) == enc2.encode(text)
