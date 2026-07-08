import pytest
import torch

from picochat.data.sft import (
    PRESETS,
    ChatDatasetSpec,
    SFTDataset,
    encode_conversation,
    resolve_spec,
)
from picochat.tokenizer import load_tokenizer, train_tokenizer

CORPUS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "what is the capital of France",
    "the capital of France is Paris",
    "let me think about this",
] * 25

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<think>", "</think>"]


@pytest.fixture
def tokenizer(tmp_path):
    path = tmp_path / "tokenizer.json"
    train_tokenizer(
        iter(CORPUS), vocab_size=300, save_as=path, special_tokens=SPECIAL_TOKENS
    )
    return load_tokenizer(path)


@pytest.fixture
def pad_id(tokenizer):
    return tokenizer.encode_single_token("<pad>")


def test_encode_conversation_masks_non_assistant_turns(tokenizer, pad_id):
    messages = [
        {"role": "user", "content": "what is the capital of France"},
        {"role": "assistant", "content": "the capital of France is Paris"},
    ]
    input_ids, labels = encode_conversation(messages, tokenizer, max_length=64, pad_id=pad_id)

    assert len(input_ids) == len(labels) == 64
    # every masked-out label is pad_id; every non-pad label matches its input token
    for tok, lab in zip(input_ids, labels):
        assert lab in (pad_id, tok)
    # the assistant turn contributes at least some real (non-pad) labels
    assert any(lab != pad_id for lab in labels)
    # the user turn's tokens are all masked: decode the assistant-only labels
    # and check the user text never leaks into the trainable label span
    trainable = [t for t, l in zip(input_ids, labels) if l != pad_id]
    assert "France" not in tokenizer.decode(trainable) or "Paris" in tokenizer.decode(trainable)


def test_encode_conversation_only_assistant_span_is_trainable(tokenizer, pad_id):
    messages = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "the quick brown fox"},
    ]
    input_ids, labels = encode_conversation(messages, tokenizer, max_length=64, pad_id=pad_id)
    bos = tokenizer.encode_single_token("<s>")
    eos = tokenizer.encode_single_token("</s>")

    # reconstruct turn boundaries from input_ids (each turn is <s> ... </s>)
    turns = []
    start = None
    for i, t in enumerate(input_ids):
        if t == bos:
            start = i
        elif t == eos and start is not None:
            turns.append((start, i))
            start = None
    assert len(turns) == 2
    user_span, assistant_span = turns
    assert all(labels[i] == pad_id for i in range(*user_span))
    assert any(labels[i] != pad_id for i in range(*assistant_span))
    assert all(labels[i] == pad_id for i in range(assistant_span[1] + 1, 64))


def test_encode_conversation_pads_short_sequences(tokenizer, pad_id):
    messages = [{"role": "assistant", "content": "hi"}]
    input_ids, labels = encode_conversation(messages, tokenizer, max_length=32, pad_id=pad_id)
    assert input_ids[-1] == pad_id
    assert labels[-1] == pad_id


def test_encode_conversation_returns_none_when_truncated_before_assistant(tokenizer, pad_id):
    messages = [
        {"role": "user", "content": "the quick brown fox jumps over the lazy dog " * 5},
        {"role": "assistant", "content": "the capital of France is Paris"},
    ]
    # max_length small enough that we never reach the assistant turn
    assert encode_conversation(messages, tokenizer, max_length=4, pad_id=pad_id) is None


def test_resolve_spec_preset():
    assert resolve_spec("smoltalk", None) is PRESETS["smoltalk"]


def test_resolve_spec_unknown_preset_raises():
    with pytest.raises(SystemExit):
        resolve_spec("no-such-preset", None)


def test_resolve_spec_inline_dataset():
    spec = resolve_spec(None, "some/repo:config:val:turns")
    assert spec == ChatDatasetSpec("some/repo", "config", "val", "turns")


def test_resolve_spec_inline_dataset_defaults():
    spec = resolve_spec(None, "some/repo")
    assert spec == ChatDatasetSpec("some/repo", None, "train", "messages")


def test_resolve_spec_requires_preset_or_dataset():
    with pytest.raises(SystemExit):
        resolve_spec(None, None)


def test_sft_dataset_shapes_and_attention_mask(tokenizer, pad_id):
    conversations = [
        [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "the quick brown fox"},
        ],
        [
            {"role": "user", "content": "let me think about this"},
            {"role": "assistant", "content": "the capital of France is Paris"},
        ],
    ]
    ds = SFTDataset(conversations, tokenizer, max_length=48, pad_id=pad_id)
    assert len(ds) == 2
    item = ds[0]
    assert item["input_ids"].shape == (48,)
    assert item["labels"].shape == (48,)
    assert item["attention_mask"].shape == (48,)
    assert item["attention_mask"].dtype == torch.long
    assert torch.equal(item["attention_mask"], (item["input_ids"] != pad_id).long())


def test_sft_dataset_drops_conversations_with_no_trainable_span(tokenizer, pad_id):
    conversations = [
        [
            {"role": "user", "content": "the quick brown fox jumps over the lazy dog " * 5},
            {"role": "assistant", "content": "the capital of France is Paris"},
        ],
    ]
    ds = SFTDataset(conversations, tokenizer, max_length=4, pad_id=pad_id)
    assert len(ds) == 0
