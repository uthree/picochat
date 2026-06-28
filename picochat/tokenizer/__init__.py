import base64
import json
import os
from typing import Iterator

import rustbpe
import tiktoken

PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"  # English contractions
    r"|\p{Han}{1,2}"  # kanji
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"  # Words (alphabetical/Unicode characters)
    r"|\p{N}"  # Digits: Always separate each digit.
    r"|\s+(?!\S)"  # trailing space
    r"|\t"  # tab
    r"|\s+"  # whitespace
)


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
