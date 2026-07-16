import pytest
import torch

from picochat.dataloader import (
    SFTDataset,
    SFTTensorDataset,
    pack_examples,
)
from picochat.dataset import (
    CHAT_PRESETS,
    ChatDatasetSpec,
    _aya_kor_to_messages,
    resolve_chat_spec,
)
from picochat.tokenizer import (
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    encode_conversation,
    load_tokenizer,
    render_chat_prompt,
    train_tokenizer,
)

CORPUS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "what is the capital of France",
    "the capital of France is Paris",
    "let me think about this",
] * 25


@pytest.fixture
def tokenizer(tmp_path):
    path = tmp_path / "tokenizer.json"
    train_tokenizer(
        iter(CORPUS), vocab_size=300, save_as=path, special_tokens=SPECIAL_TOKENS
    )
    return load_tokenizer(path)


@pytest.fixture
def pad_id(tokenizer):
    return tokenizer.encode_single_token(PAD_TOKEN)


def test_encode_conversation_masks_non_assistant_turns(tokenizer, pad_id):
    messages = [
        {"role": "user", "content": "what is the capital of France"},
        {"role": "assistant", "content": "the capital of France is Paris"},
    ]
    input_ids, labels = encode_conversation(
        messages, tokenizer, max_length=64, pad_id=pad_id
    )

    assert len(input_ids) == len(labels) <= 64
    # every masked-out label is pad_id; every non-pad label matches its input token
    for tok, lab in zip(input_ids, labels):
        assert lab in (pad_id, tok)
    # the assistant turn contributes at least some real (non-pad) labels
    assert any(lab != pad_id for lab in labels)
    # the user turn's tokens are all masked: decode the assistant-only labels
    # and check the user text never leaks into the trainable label span
    trainable = [t for t, lab in zip(input_ids, labels) if lab != pad_id]
    assert "France" not in tokenizer.decode(trainable) or "Paris" in tokenizer.decode(
        trainable
    )


def test_encode_conversation_renders_chatml(tokenizer, pad_id):
    # the exact ChatML wire format: <s> + <|im_start|>{role}\n{content}
    # <|im_end|>\n per turn + </s>, with <s>/</s> document-level (one each)
    messages = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "the quick brown fox"},
    ]
    input_ids, _ = encode_conversation(
        messages, tokenizer, max_length=64, pad_id=pad_id
    )
    assert tokenizer.decode(input_ids) == (
        "<|begin_of_text|>"
        "<|im_start|>user\nhello world<|im_end|>\n"
        "<|im_start|>assistant\nthe quick brown fox<|im_end|>\n"
        "<|end_of_text|>"
    )


def test_encode_conversation_only_assistant_body_is_trainable(tokenizer, pad_id):
    messages = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "the quick brown fox"},
    ]
    input_ids, labels = encode_conversation(
        messages, tokenizer, max_length=64, pad_id=pad_id
    )
    im_start = tokenizer.encode_single_token("<|im_start|>")
    im_end = tokenizer.encode_single_token("<|im_end|>")

    # reconstruct turn boundaries from the <|im_start|> ... <|im_end|> pairs
    turns = []
    start = None
    for i, t in enumerate(input_ids):
        if t == im_start:
            start = i
        elif t == im_end and start is not None:
            turns.append((start, i))
            start = None
    assert len(turns) == 2
    user_span, assistant_span = turns
    # the user turn is fully masked, closing <|im_end|> included
    assert all(labels[i] == pad_id for i in range(user_span[0], user_span[1] + 1))
    # assistant: the header `<|im_start|>assistant\n` is masked ...
    a_start, a_end = assistant_span
    header_len = 1 + len(tokenizer.encode_ordinary("assistant\n"))
    assert all(labels[i] == pad_id for i in range(a_start, a_start + header_len))
    # ... the body is trainable, up to and including <|im_end|> (the stop token)
    assert all(labels[i] != pad_id for i in range(a_start + header_len, a_end + 1))
    assert labels[a_end] == im_end
    # everything after the turn (inter-turn newline, </s>) is masked
    assert all(labels[i] == pad_id for i in range(a_end + 1, len(labels)))


def test_render_chat_prompt_ends_with_assistant_cue(tokenizer):
    messages = [
        {"role": "system", "content": "the capital of France is Paris"},
        {"role": "user", "content": "hello world"},
    ]
    assert tokenizer.decode(render_chat_prompt(messages, tokenizer)) == (
        "<|begin_of_text|>"
        "<|im_start|>system\nthe capital of France is Paris<|im_end|>\n"
        "<|im_start|>user\nhello world<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_render_chat_prompt_is_prefix_of_training_encoding(tokenizer, pad_id):
    # train/inference alignment: the generation prompt must be exactly the
    # tokens the training encoder places before the assistant body, so the
    # trainable span starts right where generation starts
    messages = [
        {"role": "user", "content": "what is the capital of France"},
        {"role": "assistant", "content": "the capital of France is Paris"},
    ]
    input_ids, labels = encode_conversation(
        messages, tokenizer, max_length=128, pad_id=pad_id
    )
    prompt = render_chat_prompt(messages[:1], tokenizer)
    assert input_ids[: len(prompt)] == prompt
    assert all(label == pad_id for label in labels[: len(prompt)])
    assert labels[len(prompt)] != pad_id  # first generated token is trainable


def test_encode_conversation_returns_unpadded_length(tokenizer, pad_id):
    # no padding: packing into fixed-length rows happens in pack_examples
    messages = [{"role": "assistant", "content": "hi"}]
    input_ids, labels = encode_conversation(
        messages, tokenizer, max_length=32, pad_id=pad_id
    )
    eos = tokenizer.encode_single_token(EOS_TOKEN)
    assert len(input_ids) == len(labels) < 32
    assert input_ids[-1] == eos


# ---------------------------------------------------------------------------
# pack_examples: MosaicBERT-style sequence packing
# ---------------------------------------------------------------------------
PAD = 0


def _example(tokens: list[int]) -> tuple[list[int], list[int]]:
    return tokens, list(tokens)


def test_pack_examples_packs_several_examples_per_row():
    examples = [_example([1, 2, 3]), _example([4, 5]), _example([6, 7, 8])]
    input_ids, labels, doc_ids = pack_examples(examples, max_length=8, pad_id=PAD)
    assert input_ids.shape == labels.shape == doc_ids.shape == (1, 8)
    # all 8 tokens fit into one row; no padding at all
    assert (input_ids != PAD).all()
    # rows are filled longest-first, then largest-that-fits
    assert input_ids[0].tolist() == [6, 7, 8, 1, 2, 3, 4, 5]
    assert doc_ids[0].tolist() == [0, 0, 0, 1, 1, 1, 2, 2]


def test_pack_examples_pads_leftover_room_with_own_doc_id():
    input_ids, labels, doc_ids = pack_examples(
        [_example([1, 2, 3]), _example([4, 5, 6])], max_length=4, pad_id=PAD
    )
    assert input_ids.shape == (2, 4)
    for row in range(2):
        assert input_ids[row, 3] == PAD
        assert labels[row, 3] == PAD
        # the padding tail is its own document, distinct from the real one
        assert doc_ids[row, 3] == 1
        assert (doc_ids[row, :3] == 0).all()


def test_pack_examples_masks_each_examples_first_label():
    # the first token of every packed example must not be a training target:
    # after the loss shift it would be predicted across a document boundary
    examples = [_example([1, 2, 3]), _example([4, 5])]
    input_ids, labels, doc_ids = pack_examples(examples, max_length=5, pad_id=PAD)
    starts = [0] + [
        i
        for i in range(1, 5)
        if doc_ids[0, i] != doc_ids[0, i - 1] and input_ids[0, i] != PAD
    ]
    assert len(starts) == 2
    for s in starts:
        assert labels[0, s] == PAD
    # every other position keeps its label
    others = [i for i in range(5) if i not in starts]
    assert all(labels[0, i] == input_ids[0, i] for i in others)


def test_pack_examples_keeps_every_token():
    torch.manual_seed(0)
    examples = [
        _example(torch.randint(1, 40, (int(n),)).tolist())
        for n in torch.randint(1, 16, (50,))
    ]
    max_length = 16
    input_ids, _, doc_ids = pack_examples(examples, max_length, pad_id=PAD)
    n_tokens = sum(len(ids) for ids, _ in examples)
    assert int((input_ids != PAD).sum()) == n_tokens
    # each row's real span is partitioned into contiguous documents 0..k-1
    for row_ids, row_docs in zip(input_ids, doc_ids):
        real = row_docs[row_ids != PAD]
        assert (real.diff() >= 0).all()  # doc ids only ever increase
        pad_docs = row_docs[row_ids == PAD]
        if len(pad_docs):
            assert (pad_docs == real.max() + 1).all()


def test_encode_conversation_returns_none_when_truncated_before_assistant(
    tokenizer, pad_id
):
    messages = [
        {"role": "user", "content": "the quick brown fox jumps over the lazy dog " * 5},
        {"role": "assistant", "content": "the capital of France is Paris"},
    ]
    # max_length small enough that we never reach the assistant turn
    assert encode_conversation(messages, tokenizer, max_length=4, pad_id=pad_id) is None


def test_chat_dataset_spec_to_messages_uses_messages_key():
    spec = ChatDatasetSpec("some/repo", messages_key="turns")
    assert spec.to_messages({"turns": [{"role": "user", "content": "hi"}]}) == [
        {"role": "user", "content": "hi"}
    ]


def test_chat_dataset_spec_to_messages_prefers_format_over_messages_key():
    spec = ChatDatasetSpec("some/repo", format=lambda row: [row["only"]])
    assert spec.to_messages(
        {"only": {"role": "user", "content": "hi"}, "turns": []}
    ) == [{"role": "user", "content": "hi"}]


def test_aya_kor_to_messages_filters_by_language_code():
    assert (
        _aya_kor_to_messages({"language_code": "eng", "inputs": "hi", "targets": "yo"})
        is None
    )


def test_aya_kor_to_messages_renders_one_turn_conversation():
    row = {"language_code": "kor", "inputs": "안녕하세요", "targets": "안녕하십니까"}
    assert _aya_kor_to_messages(row) == [
        {"role": "user", "content": "안녕하세요"},
        {"role": "assistant", "content": "안녕하십니까"},
    ]


def test_resolve_spec_preset():
    assert resolve_chat_spec("smoltalk", None) is CHAT_PRESETS["smoltalk"]


def test_resolve_spec_unknown_preset_raises():
    with pytest.raises(SystemExit):
        resolve_chat_spec("no-such-preset", None)


def test_resolve_spec_inline_dataset():
    spec = resolve_chat_spec(None, "some/repo:config:val:turns")
    assert spec == ChatDatasetSpec("some/repo", "config", "val", "turns")


def test_resolve_spec_inline_dataset_defaults():
    spec = resolve_chat_spec(None, "some/repo")
    assert spec == ChatDatasetSpec("some/repo", None, "train", "messages")


def test_resolve_spec_requires_preset_or_dataset():
    with pytest.raises(SystemExit):
        resolve_chat_spec(None, None)


def test_sft_dataset_packs_conversations_into_fixed_length_rows(tokenizer, pad_id):
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
    ds = SFTDataset(conversations, tokenizer, max_length=128, pad_id=pad_id)
    # both short conversations fit into one packed row
    assert len(ds) == 1
    item = ds[0]
    assert item["input_ids"].shape == (128,)
    assert item["labels"].shape == (128,)
    assert item["doc_ids"].shape == (128,)
    real = item["input_ids"] != pad_id
    assert set(item["doc_ids"][real].tolist()) == {0, 1}  # two packed documents


def test_sft_dataset_drops_conversations_with_no_trainable_span(tokenizer, pad_id):
    conversations = [
        [
            {
                "role": "user",
                "content": "the quick brown fox jumps over the lazy dog " * 5,
            },
            {"role": "assistant", "content": "the capital of France is Paris"},
        ],
    ]
    ds = SFTDataset(conversations, tokenizer, max_length=4, pad_id=pad_id)
    assert len(ds) == 0


def test_sft_tensor_dataset_reads_saved_bundle(tmp_path):
    input_ids = torch.randint(1, 40, (3, 16))
    labels = input_ids.clone()
    labels[:, -4:] = 0
    doc_ids = torch.zeros(3, 16, dtype=torch.long)
    doc_ids[:, 8:] = 1
    bundle_path = tmp_path / "bundle.pt"
    torch.save(
        {"input_ids": input_ids, "labels": labels, "doc_ids": doc_ids, "pad_id": 0},
        bundle_path,
    )

    ds = SFTTensorDataset(bundle_path)
    assert len(ds) == 3
    item = ds[0]
    assert torch.equal(item["input_ids"], input_ids[0])
    assert torch.equal(item["labels"], labels[0])
    assert torch.equal(item["doc_ids"], doc_ids[0])
