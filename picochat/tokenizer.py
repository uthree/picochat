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

# Audio (multimodal input), Qwen2-Audio-style: an audio clip is rendered as
# AUDIO_BOS + AUDIO_TOKEN * n + AUDIO_EOS, where AUDIO_TOKEN is a placeholder
# whose embedding is replaced at runtime by the audio encoder's soft tokens
# (see picochat.audio). n = number of ~100ms audio tokens for the clip.
AUDIO_BOS_TOKEN = "<|audio_bos|>"
AUDIO_TOKEN = "<|AUDIO|>"
AUDIO_EOS_TOKEN = "<|audio_eos|>"

# Image (multimodal input), Qwen2-VL-style: an image is rendered as
# VISION_START + IMAGE_TOKEN * n + VISION_END, IMAGE_TOKEN being the placeholder
# whose embedding the vision encoder replaces at runtime. Only the tokens are
# reserved here; the image encoder itself is future work (audio ships first).
VISION_START_TOKEN = "<|vision_start|>"
IMAGE_TOKEN = "<|image_pad|>"
VISION_END_TOKEN = "<|vision_end|>"

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
    AUDIO_BOS_TOKEN,
    AUDIO_TOKEN,
    AUDIO_EOS_TOKEN,
    VISION_START_TOKEN,
    IMAGE_TOKEN,
    VISION_END_TOKEN,
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


# ---------------------------------------------------------------------------
# ChatML rendering (nanochat keeps the chat format next to the tokenizer too:
# the special tokens above are baked into it, so the rendering that uses them
# lives here and training, preprocessing and inference can never drift).
#
# Conversations are rendered in ChatML, the de-facto standard chat format of
# modern open-weight models (Qwen, SmolLM, ...):
#
#     <|begin_of_text|><|im_start|>{role}\n{content}<|im_end|>\n ... <|end_of_text|>
#
# `<|begin_of_text|>`/`<|end_of_text|>` stay document-level delimiters, exactly
# as in the pretraining corpus (one per conversation, not per turn), while
# `<|im_start|>`/`<|im_end|>` delimit turns; `<|im_end|>` is the chat stop
# token at inference. Turns are tokenized turn-by-turn (not as one joined
# string) so token spans line up exactly with turn boundaries -- role and
# content are encoded with encode_ordinary, so they can never resolve to a
# special token even if they contain text like "<|im_end|>". Reasoning traces
# (<think>...</think>) are just ordinary assistant content -- nothing here
# special-cases them.
# ---------------------------------------------------------------------------


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
    sequences happens later (see picochat.dataloader.pack_examples).

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


def train_tokenizer(
    text_iterator: Iterator[str],
    vocab_size: int = 32000,
    name: str = "tokenizer",
    save_as: os.PathLike = "tokenizer.json",
    special_tokens: list[str] = SPECIAL_TOKENS,
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
