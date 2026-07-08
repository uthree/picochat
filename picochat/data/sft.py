"""SFT (supervised fine-tuning) chat data: turn HF conversation datasets
(e.g. HuggingFaceTB/smoltalk) into padded (input_ids, labels, attention_mask)
tensors for GPT training.

Each conversation is rendered turn-by-turn as `<s>{role}\\n{content}</s>` and
tokenized turn-by-turn (not as one joined string) so token spans line up
exactly with turn boundaries -- a message's content is encoded with
encode_ordinary, so it can never resolve to a special token even if it
contains text like "<s>". Only assistant turns contribute to the loss: every
other position's label is set to the tokenizer's `<pad>` id, the same id GPT
already treats as ignore_index in its cross-entropy loss (see
GPT._head_loss), so no separate -100 convention is needed. Reasoning traces
(<think>...</think>) are just ordinary assistant content -- nothing here
special-cases them.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tiktoken
import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class ChatDatasetSpec:
    path: str  # HF Hub repo id, e.g. "HuggingFaceTB/smoltalk"
    name: str | None = None  # config / subset name, e.g. "all"
    split: str = "train"
    messages_key: str = "messages"  # column holding the list of {role, content}


PRESETS: dict[str, ChatDatasetSpec] = {
    "smoltalk": ChatDatasetSpec("HuggingFaceTB/smoltalk", "all"),
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
        messages = row[spec.messages_key]
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


def encode_conversation(
    messages: list[dict],
    tokenizer: tiktoken.Encoding,
    max_length: int,
    pad_id: int,
) -> tuple[list[int], list[int]] | None:
    """Tokenize one conversation into (input_ids, labels), both length
    max_length. Non-assistant turns and right-padding get `pad_id` in labels.
    Returns None if truncation left no assistant turn to train on.
    """
    bos = tokenizer.encode_single_token("<s>")
    eos = tokenizer.encode_single_token("</s>")
    input_ids: list[int] = []
    labels: list[int] = []
    for msg in messages:
        body = tokenizer.encode_ordinary(f"{msg['role']}\n{msg['content']}")
        turn = [bos, *body, eos]
        input_ids.extend(turn)
        is_assistant = msg["role"] == "assistant"
        labels.extend(turn if is_assistant else [pad_id] * len(turn))
        if len(input_ids) >= max_length:
            break

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    if all(label == pad_id for label in labels):
        return None  # nothing survived truncation to train on

    pad_amount = max_length - len(input_ids)
    input_ids.extend([pad_id] * pad_amount)
    labels.extend([pad_id] * pad_amount)
    return input_ids, labels


class SFTDataset(Dataset):
    """Pre-tokenizes every conversation once at construction time (SFT corpora
    fit in memory, unlike pretraining's token-stream shards) into fixed-length,
    right-padded (input_ids, labels, attention_mask) triples."""

    def __init__(
        self,
        conversations: list[list[dict]],
        tokenizer: tiktoken.Encoding,
        max_length: int,
        pad_id: int,
    ):
        self.pad_id = pad_id
        self.examples: list[tuple[list[int], list[int]]] = [
            encoded
            for messages in conversations
            if (encoded := encode_conversation(messages, tokenizer, max_length, pad_id))
            is not None
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        input_ids, labels = self.examples[idx]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class SFTTensorDataset(Dataset):
    """Reads a (input_ids, labels) tensor bundle written by
    scripts/sft_setup.py's `process()` -- the on-disk counterpart of
    SFTDataset, for training runs that shouldn't re-tokenize on every launch.
    """

    def __init__(self, path: str | Path):
        bundle = torch.load(path, map_location="cpu")
        self.input_ids = bundle["input_ids"]
        self.labels = bundle["labels"]
        self.pad_id = bundle["pad_id"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}
