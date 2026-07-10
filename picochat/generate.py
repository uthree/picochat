"""Sampling and incremental (KV-cached) generation for chat/serving code.

SamplingConfig carries the knobs the chat TUI exposes via `/set`; its
`update()` accepts raw strings so the same validation backs CLI flags and
interactive commands. `generate()` is a lazy token iterator -- callers stream
tokens as they arrive and can simply stop consuming it to abort generation.
"""

from dataclasses import dataclass
from typing import Iterator

import torch
from tiktoken import Encoding
from torch import Tensor

from picochat.tokenizer import EOS_TOKEN, IM_END


@dataclass
class SamplingConfig:
    temperature: float = 0.8  # 0 -> greedy decoding
    top_k: int | None = 50  # None -> disabled
    top_p: float | None = None  # nucleus sampling; None -> disabled
    max_new_tokens: int = 256

    def update(self, key: str, raw: str) -> None:
        """Set one field from a raw string (e.g. `/set top_k 50`); "none"/
        "off" disables the optional filters. Raises ValueError on bad input."""
        try:
            if key == "temperature":
                value = float(raw)
                if value < 0:
                    raise ValueError
                self.temperature = value
            elif key == "top_k":
                top_k = None if raw.lower() in ("none", "off") else int(raw)
                if top_k is not None and top_k < 1:
                    raise ValueError
                self.top_k = top_k
            elif key == "top_p":
                top_p = None if raw.lower() in ("none", "off") else float(raw)
                if top_p is not None and not 0 < top_p <= 1:
                    raise ValueError
                self.top_p = top_p
            elif key == "max_new_tokens":
                value = int(raw)
                if value < 1:
                    raise ValueError
                self.max_new_tokens = value
            else:
                raise ValueError(
                    f"unknown setting '{key}' (temperature, top_k, top_p, "
                    "max_new_tokens)"
                )
        except ValueError as e:
            if str(e):
                raise
            raise ValueError(f"invalid value '{raw}' for {key}") from None

    def describe(self) -> str:
        return (
            f"temperature={self.temperature:g}  "
            f"top_k={self.top_k if self.top_k is not None else 'off'}  "
            f"top_p={self.top_p if self.top_p is not None else 'off'}  "
            f"max_new_tokens={self.max_new_tokens}"
        )


def sample(logits: Tensor, cfg: SamplingConfig) -> Tensor:
    """logits: (B, V) -> next token ids (B, 1). top-k then top-p filtering,
    both optional; temperature <= 0 short-circuits to greedy."""
    if cfg.temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / cfg.temperature
    if cfg.top_k is not None and cfg.top_k < logits.shape[-1]:
        threshold = torch.topk(logits, cfg.top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if cfg.top_p is not None and cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Drop tokens once the mass *before* them already reaches top_p, so
        # the most probable token always survives.
        drop = (cum - torch.softmax(sorted_logits, dim=-1)) >= cfg.top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: Encoding,
    prompt_ids: list[int],
    cfg: SamplingConfig,
    device: torch.device | str = "cpu",
) -> Iterator[int]:
    """Stream token ids continuing `prompt_ids` (KV-cached decode) until
    `<|im_end|>` (the ChatML stop token), `<|end_of_text|>`, or the token
    budget. `model` is a TransformerLM (e.g. gpt.model). Lazy: breaking out
    of the loop aborts generation immediately."""
    stop_ids = {
        tokenizer.encode_single_token(IM_END),
        tokenizer.encode_single_token(EOS_TOKEN),
    }

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, cache, pos = model.decode(x)
    next_token = sample(logits[:, -1], cfg)

    for _ in range(cfg.max_new_tokens):
        token_id = int(next_token.item())
        if token_id in stop_ids:
            return
        yield token_id
        logits, cache, pos = model.decode(next_token, cache, pos)
        next_token = sample(logits[:, -1], cfg)
