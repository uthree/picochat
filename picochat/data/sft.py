"""SFT (supervised fine-tuning) chat data: turn HF conversation datasets
(e.g. HuggingFaceTB/smoltalk) into packed (input_ids, labels, doc_ids)
tensors for SFT training.

Conversations are rendered in ChatML, the de-facto standard chat format of
modern open-weight models (Qwen, SmolLM, ...):

    <|begin_of_text|><|im_start|>{role}\\n{content}<|im_end|>\\n ... <|end_of_text|>

`<|begin_of_text|>`/`<|end_of_text|>` stay document-level delimiters, exactly
as in the pretraining corpus (one per conversation, not per turn), while
`<|im_start|>`/`<|im_end|>`
delimit turns; `<|im_end|>` is the chat stop token at inference. Turns are
tokenized turn-by-turn (not as one joined string) so token spans line up
exactly with turn boundaries -- role and content are encoded with
encode_ordinary, so they can never resolve to a special token even if they
contain text like "<|im_end|>". Only assistant turns contribute to the loss,
and within them only the content plus the closing `<|im_end|>` (the model
must learn to emit it to stop): every other position's label is set to the
tokenizer's `<|pad|>` id, the same id the loss already treats as ignore_index
(see SFTModule._loss), so no separate -100 convention is needed. Reasoning
traces (<think>...</think>) are just ordinary assistant content -- nothing
here special-cases them.

Instead of padding each conversation to max_length on its own, several
conversations are packed into one fixed-length sequence (MosaicBERT-style
sequence packing, see pack_examples); the per-token doc_ids let attention be
confined to each conversation (see Transformer.forward).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import tiktoken
import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset

from picochat.data.packing import pack_bins
from picochat.tokenizer import BOS_TOKEN, EOS_TOKEN, IM_END, IM_START


@dataclass
class ChatDatasetSpec:
    path: str  # HF Hub repo id, e.g. "HuggingFaceTB/smoltalk"
    name: str | None = None  # config / subset name, e.g. "all"
    split: str = "train"
    messages_key: str = "messages"  # column holding the list of {role, content}
    # For sources that aren't already a {role, content} list (e.g. single-turn
    # inputs/targets columns, or a language-tagged corpus to filter): maps a
    # raw row to a messages list, or None to skip the row. Overrides
    # messages_key. Mirrors picochat.data.pretrain.DatasetSpec.format.
    format: Callable[[dict], list[dict] | None] | None = field(default=None, repr=False)

    def to_messages(self, row: dict) -> list[dict] | None:
        if self.format is not None:
            return self.format(row)
        return row[self.messages_key]


def _aya_kor_to_messages(row: dict) -> list[dict] | None:
    """CohereLabs/aya_dataset is single-turn instruction/response pairs across
    65 languages in one split (inputs/targets/language_code); filter down to
    Korean and reshape into a one-turn conversation; swapping the
    `language_code` ("kor") covers the dataset's other 64 languages with the
    same shape."""
    if row["language_code"] != "kor":
        return None
    return [
        {"role": "user", "content": row["inputs"]},
        {"role": "assistant", "content": row["targets"]},
    ]


PRESETS: dict[str, ChatDatasetSpec] = {
    "smoltalk": ChatDatasetSpec("HuggingFaceTB/smoltalk", "all"),
    # Smoltalk's prompts machine-translated into 8 languages (respecting
    # local conventions), then answered natively in each target language.
    "smoltalk-multilingual": ChatDatasetSpec(
        "HuggingFaceTB/smoltalk2",
        "SFT",
        split="smoltalk_multilingual_8languages_lang_5_no_think",
    ),  # es/fr/it/pt/de/ar/ru/zh, ~254k rows
    # Magpie-style Japanese: CALM3-22B-Chat generates the user turns,
    # Qwen2.5-32B-Instruct the assistant responses.
    "magpie-ja": ChatDatasetSpec(
        "llm-jp/magpie-sft-v1.0", messages_key="conversations"
    ),  # ~132k rows
    # Human-annotated Korean slice of CohereLabs/aya_dataset (see
    # _aya_kor_to_messages) -- smaller and single-turn, but not
    # machine-translated like most other open Korean instruction sets.
    "aya-ko": ChatDatasetSpec("CohereLabs/aya_dataset", format=_aya_kor_to_messages),
}


def iter_conversations(
    spec: ChatDatasetSpec,
    streaming: bool = True,
    limit: int | None = None,
) -> Iterator[list[dict]]:
    """Yield one conversation (a list of {"role", "content"} dicts) at a time."""
    ds = load_dataset(spec.path, spec.name, split=spec.split, streaming=streaming)
    if limit is not None:
        ds = ds.take(limit) if streaming else ds.select(range(min(limit, len(ds))))
    for row in ds:
        messages = spec.to_messages(row)
        if messages:
            yield messages


def resolve_spec(preset: str | None, dataset: str | None) -> ChatDatasetSpec:
    """Resolve a ChatDatasetSpec from CLI arguments.

    Either --preset <name> or --dataset "path[:name[:split[:messages_key]]]".
    """
    if preset is not None:
        if preset not in PRESETS:
            raise SystemExit(
                f"unknown preset '{preset}'. choices: {', '.join(PRESETS)}"
            )
        return PRESETS[preset]
    if dataset is not None:
        path, *rest = dataset.split(":")
        name = rest[0] if len(rest) > 0 and rest[0] else None
        split = rest[1] if len(rest) > 1 and rest[1] else "train"
        messages_key = rest[2] if len(rest) > 2 and rest[2] else "messages"
        return ChatDatasetSpec(path, name, split, messages_key)
    raise SystemExit("either --preset or --dataset is required")


def render_turn(
    role: str, content: str, tokenizer: tiktoken.Encoding
) -> tuple[list[int], list[int], list[int]]:
    """One ChatML turn `<|im_start|>{role}\\n{content}<|im_end|>\\n` as three
    token spans: (header, body, tail) = (`<|im_start|>{role}\\n`,
    `{content}<|im_end|>`, `\\n`). Split this way because they carry different
    loss masks in encode_conversation, and the header doubles as the
    generation cue in render_chat_prompt."""
    im_start = tokenizer.encode_single_token(IM_START)
    im_end = tokenizer.encode_single_token(IM_END)
    header = [im_start, *tokenizer.encode_ordinary(f"{role}\n")]
    body = [*tokenizer.encode_ordinary(content), im_end]
    tail = tokenizer.encode_ordinary("\n")
    return header, body, tail


def render_chat_prompt(
    messages: list[dict],
    tokenizer: tiktoken.Encoding,
) -> list[int]:
    """Token ids of a conversation prompt ready for generation:
    `<|begin_of_text|>`, every turn so far, then the bare assistant header
    `<|im_start|>assistant\\n` to
    cue the reply. The model continues with the assistant body and stops at
    `<|im_end|>` -- the exact spans encode_conversation trains. An optional
    system prompt is just a leading {"role": "system", ...} message."""
    ids = [tokenizer.encode_single_token(BOS_TOKEN)]
    for msg in messages:
        header, body, tail = render_turn(msg["role"], msg["content"], tokenizer)
        ids.extend(header + body + tail)
    header, _, _ = render_turn("assistant", "", tokenizer)
    ids.extend(header)
    return ids


def encode_conversation(
    messages: list[dict],
    tokenizer: tiktoken.Encoding,
    max_length: int,
    pad_id: int,
) -> tuple[list[int], list[int]] | None:
    """Tokenize one conversation in ChatML into (input_ids, labels) of equal,
    variable length <= max_length -- no padding; packing into fixed-length
    sequences happens later (see pack_examples).

    Only assistant turn *bodies* (content + `<|im_end|>`) are trainable; turn
    headers (`<|im_start|>{role}\\n`), the inter-turn newline and the
    document-level `<|begin_of_text|>`/`<|end_of_text|>` get `pad_id` labels
    -- at inference the header is part of the generation prompt and decoding
    stops at `<|im_end|>`, so
    none of them are the model's to produce. Returns None if truncation left
    no assistant turn to train on.
    """
    bos = tokenizer.encode_single_token(BOS_TOKEN)
    eos = tokenizer.encode_single_token(EOS_TOKEN)
    input_ids: list[int] = [bos]
    labels: list[int] = [pad_id]
    for msg in messages:
        header, body, tail = render_turn(msg["role"], msg["content"], tokenizer)
        input_ids.extend(header + body + tail)
        is_assistant = msg["role"] == "assistant"
        labels.extend([pad_id] * len(header))
        labels.extend(body if is_assistant else [pad_id] * len(body))
        labels.extend([pad_id] * len(tail))
        if len(input_ids) >= max_length:
            break
    else:  # untruncated: close the document like the pretraining corpus does
        input_ids.append(eos)
        labels.append(pad_id)

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    if all(label == pad_id for label in labels):
        return None  # nothing survived truncation to train on
    return input_ids, labels


def pack_examples(
    examples: list[tuple[list[int], list[int]]],
    max_length: int,
    pad_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack variable-length (input_ids, labels) examples into fixed-length
    sequences, several examples per sequence, instead of padding each one to
    max_length on its own (MosaicBERT-style sequence packing, see
    picochat.data.packing.pack_bins for the bin-assignment algorithm). Room
    that nothing fits into is padded with pad_id.

    Returns (input_ids, labels, doc_ids), each (n_bins, max_length) int64.
    doc_ids numbers the examples within a bin, with the padding tail getting
    its own final id, so attention can be confined to one example (see
    Transformer.forward). Each example's first-token label is forced to pad_id:
    after the loss shift it would be predicted from the *previous* example's
    last token, which the document mask hides (at a sequence start it was
    never a target to begin with).
    """
    bins = pack_bins([len(ids) for ids, _ in examples], max_length)

    input_ids = torch.full((len(bins), max_length), pad_id, dtype=torch.long)
    labels = torch.full_like(input_ids, pad_id)
    doc_ids = torch.zeros_like(input_ids)
    for b, packed in enumerate(bins):
        pos = 0
        for d, idx in enumerate(packed):
            ids, labs = examples[idx]
            end = pos + len(ids)
            input_ids[b, pos:end] = torch.tensor(ids)
            labels[b, pos:end] = torch.tensor(labs)
            labels[b, pos] = pad_id  # never a cross-example target (see above)
            doc_ids[b, pos:end] = d
            pos = end
        doc_ids[b, pos:] = len(packed)  # padding tail: its own document
    return input_ids, labels, doc_ids


class SFTDataset(Dataset):
    """Pre-tokenizes and packs every conversation once at construction time
    (SFT corpora fit in memory, unlike pretraining's token-stream shards) into
    fixed-length (input_ids, labels, doc_ids) sequences, several conversations
    per sequence (see pack_examples)."""

    def __init__(
        self,
        conversations: list[list[dict]],
        tokenizer: tiktoken.Encoding,
        max_length: int,
        pad_id: int,
    ):
        self.pad_id = pad_id
        examples = [
            encoded
            for messages in conversations
            if (encoded := encode_conversation(messages, tokenizer, max_length, pad_id))
            is not None
        ]
        self.input_ids, self.labels, self.doc_ids = pack_examples(
            examples, max_length, pad_id
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "doc_ids": self.doc_ids[idx],
        }


class SFTTensorDataset(Dataset):
    """Reads a (input_ids, labels, doc_ids) tensor bundle written by
    scripts/sft_setup.py's `process()` -- the on-disk counterpart of
    SFTDataset, for training runs that shouldn't re-tokenize on every launch.
    """

    def __init__(self, path: str | Path):
        bundle = torch.load(path, map_location="cpu")
        self.input_ids = bundle["input_ids"]
        self.labels = bundle["labels"]
        self.doc_ids = bundle["doc_ids"]
        self.pad_id = bundle["pad_id"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "doc_ids": self.doc_ids[idx],
        }
