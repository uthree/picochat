import base64
import json
import os
from typing import Iterator

import rustbpe
import tiktoken

# GPT-2's pattern with one change: split digits one at a time (\p{N}+ -> \p{N}).
PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d"  # English contractions
    r"| ?\p{L}+"  # words (any script)
    r"| ?\p{N}"  # Digits: Always separate each digit.
    r"| ?[^\s\p{L}\p{N}]+"  # runs of punctuation/symbols
    r"|\s+(?!\S)"  # trailing space
    r"|\s+"  # whitespace
)

# Special tokens, ChatML-style `<|...|>` notation throughout. Defined here --
# next to the tokenizer they are baked into -- so training, preprocessing and
# inference all reference one definition and can never drift.
PAD_TOKEN = "<|pad|>"  # loss ignore-index / packing filler
BOS_TOKEN = (
    "<|begin_of_text|>"  # start of a document (pretraining) / conversation (SFT)
)
EOS_TOKEN = "<|end_of_text|>"  # end of a document / conversation
IM_START = "<|im_start|>"  # ChatML: start of a turn (followed by "{role}\n")
IM_END = "<|im_end|>"  # ChatML: end of a turn (the chat stop token)

NUM_RESERVED_SPECIAL_TOKENS = 16
SPECIAL_TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    IM_START,
    IM_END,
    # Reasoning-trace delimiters, deliberately NOT pipe-style: `<think>` is
    # the de-facto spelling of Qwen3 / DeepSeek-R1. Unused by the current
    # code (assistant content is encoded with encode_ordinary), reserved for
    # reasoning training.
    "<think>",
    "</think>",
] + [f"<|reserved_token_{n}|>" for n in range(NUM_RESERVED_SPECIAL_TOKENS)]


def save_tokenizer(enc: tiktoken.Encoding, path: os.PathLike):
    data = {
        "pattern": enc._pat_str,
        "special_tokens": enc._special_tokens,
        "mergeable_ranks": {
            base64.b64encode(k).decode(): v for k, v in enc._mergeable_ranks.items()
        },
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_tokenizer(path: os.PathLike, name: str = "tokenizer") -> tiktoken.Encoding:
    with open(path) as f:
        data = json.load(f)

    return tiktoken.Encoding(
        name=name,
        pat_str=data["pattern"],
        mergeable_ranks={
            base64.b64decode(k): v for k, v in data["mergeable_ranks"].items()
        },
        special_tokens=data["special_tokens"],
    )


def train_tokenizer(
    text_iterator: Iterator[str],
    vocab_size: int = 32000,
    name: str = "tokenizer",
    save_as: os.PathLike = "tokenizer.json",
    special_tokens: list[str] | None = None,
):
    tokenizer = rustbpe.Tokenizer()
    tokenizer.train_from_iterator(
        text_iterator,
        vocab_size=vocab_size - len(special_tokens),
        pattern=PATTERN,
    )
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    special_token_map = {}
    if special_tokens:
        next_id = len(mergeable_ranks)
        for token in special_tokens:
            special_token_map[token] = next_id
            next_id += 1
    enc = tiktoken.Encoding(
        name=name,
        pat_str=PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_token_map,
    )
    save_tokenizer(enc, save_as)
