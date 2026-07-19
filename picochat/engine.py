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
    max_seq_len: int | None = None,
) -> Iterator[int]:
    """Stream token ids continuing `prompt_ids` (KV-cached decode) until
    `<|im_end|>` (the ChatML stop token), `<|end_of_text|>`, or the token
    budget. `model` is a TransformerLM (e.g. gpt.model). Lazy: breaking out
    of the loop aborts generation immediately.

    `max_seq_len` is the model's positional range (its RoPE tables): decoding
    past it asserts, so the budget is capped to stop generation early instead.
    The caller is responsible for a prompt that already fits (see
    ChatApp._build_prompt / picochat.tasks.encode_choice).

    Greedy decoding (temperature <= 0) on a model with MTP heads routes through
    generate_speculative: the emitted stream is identical (every draft is
    verified against the model's own next-token argmax), it just arrives in
    fewer forwards -- so every caller gets the speedup for free."""
    if cfg.temperature <= 0 and getattr(model, "n_mtp", 0) > 0:
        yield from generate_speculative(
            model, tokenizer, prompt_ids, cfg, device, max_seq_len
        )
        return

    budget = cfg.max_new_tokens
    if max_seq_len is not None:
        budget = min(budget, max_seq_len - len(prompt_ids))
        if budget <= 0:
            return

    stop_ids = {
        tokenizer.encode_single_token(IM_END),
        tokenizer.encode_single_token(EOS_TOKEN),
    }

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, cache, pos = model.decode(x)
    next_token = sample(logits[:, -1], cfg)

    for _ in range(budget):
        token_id = int(next_token.item())
        if token_id in stop_ids:
            return
        yield token_id
        logits, cache, pos = model.decode(next_token, cache, pos)
        next_token = sample(logits[:, -1], cfg)


def _truncate_cache(cache: list[Tensor], drop: int) -> list[Tensor]:
    """Drop the last `drop` positions from every layer's KV cache. The rejected
    speculative drafts are always the most recently appended entries, so this
    rolls the cache back to the last accepted token (works for both the full and
    the sliding-window layers, whose caches all grew by the same chunk length)."""
    if drop <= 0:
        return cache
    return [c[..., : c.shape[-2] - drop, :] for c in cache]


@torch.no_grad()
def generate_speculative(
    model: torch.nn.Module,
    tokenizer: Encoding,
    prompt_ids: list[int],
    cfg: SamplingConfig,
    device: torch.device | str = "cpu",
    max_seq_len: int | None = None,
) -> Iterator[int]:
    """Greedy self-speculative decoding driven by the model's own MTP heads.

    Emits exactly the greedy (argmax) token stream -- identical to what
    generate() would produce at temperature 0 -- but each model forward drafts
    n_mtp extra tokens (primary head -> next token, MTP head j -> the token 2+j
    ahead) and *verifies* the whole draft in one chunked forward, accepting the
    longest correct prefix. When the drafts are right this advances several
    tokens per forward; when they are wrong it still advances one, so the output
    never differs from plain greedy decoding.

    Requires model.n_mtp >= 1; otherwise falls back to plain generate(). Sampling
    knobs in `cfg` are ignored (this path is greedy); use generate() for sampled
    decoding.
    """
    k = getattr(model, "n_mtp", 0)
    if k == 0:
        yield from generate(model, tokenizer, prompt_ids, cfg, device, max_seq_len)
        return

    budget = cfg.max_new_tokens
    if max_seq_len is not None:
        budget = min(budget, max_seq_len - len(prompt_ids))
        if budget <= 0:
            return
    stop_ids = {
        tokenizer.encode_single_token(IM_END),
        tokenizer.encode_single_token(EOS_TOKEN),
    }

    # `cache_margin=k` keeps k extra keys per windowed layer so the cache can be
    # rolled back past up to k rejected drafts and still hold a full window.
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, mtp, cache, pos = model.decode_heads(x, cache_margin=k)
    cand = [int(logits[0, -1].argmax())] + [int(m[0, -1].argmax()) for m in mtp]

    emitted = 0
    while emitted < budget:
        if max_seq_len is not None:  # keep the verify chunk within the RoPE range
            cand = cand[: max_seq_len - pos]
            if not cand:
                return
        ct = torch.tensor([cand], dtype=torch.long, device=device)
        logits, mtp, cache, pos = model.decode_heads(ct, cache, pos, cache_margin=k)
        # true[i] = the model's real next token after candidate i.
        true = [int(logits[0, i].argmax()) for i in range(len(cand))]
        # cand[0] is a verified token; accept drafts while each matches the real
        # next token of its predecessor.
        accepted = 0
        for i in range(1, len(cand)):
            if cand[i] == true[i - 1]:
                accepted = i
            else:
                break
        for i in range(accepted + 1):
            tok = cand[i]
            if tok in stop_ids:
                return
            yield tok
            emitted += 1
            if emitted >= budget:
                return
        # Roll the cache back past the rejected drafts, then draft afresh: the
        # correction true[accepted] is the next verified token, and the MTP heads
        # read at the accepted position propose its successors.
        cache = _truncate_cache(cache, (len(cand) - 1) - accepted)
        pos -= (len(cand) - 1) - accepted
        cand = [true[accepted]] + [int(mtp[j][0, accepted].argmax()) for j in range(k)]


def resolve_device(spec: str | None) -> torch.device:
    """Pick a torch device: honor an explicit spec (e.g. "cuda:1", "cpu"),
    otherwise prefer CUDA, then Apple MPS, then CPU. Shared by the inference
    CLIs (chat, api, base_eval)."""
    if spec:
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def add_sampling_args(parser, *, temperature: float = 0.8, temp_help: str = "0 -> greedy decoding") -> None:
    """Add the shared --temperature/--top-k/--top-p/--max-new-tokens flags to an
    argparse parser (used by chat.py and api.py); pair with sampling_from_args."""
    parser.add_argument("--temperature", type=float, default=temperature, help=temp_help)
    parser.add_argument("--top-k", type=int, default=50, help="0 -> disabled")
    parser.add_argument("--top-p", type=float, default=1.0, help="1.0 -> disabled")
    parser.add_argument("--max-new-tokens", type=int, default=256)


def sampling_from_args(args) -> SamplingConfig:
    """Build a SamplingConfig from add_sampling_args' parsed flags, applying the
    disabling conventions (top_k 0 -> None, top_p outside (0, 1) -> None)."""
    return SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if 0 < args.top_p < 1 else None,
        max_new_tokens=args.max_new_tokens,
    )
