"""Sampling and incremental (KV-cached) generation for chat/serving code.

SamplingConfig carries the knobs the chat TUI exposes via `/set`; its
`update()` accepts raw strings so the same validation backs CLI flags and
interactive commands. `generate()` is a lazy token iterator -- callers stream
tokens as they arrive and can simply stop consuming it to abort generation.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor

from picochat.tokenizer import EOS_TOKEN, IM_END, Tokenizer as Encoding


def inference_autocast(model: torch.nn.Module, device: torch.device | str):
    """The autocast context inference should run under: bf16/fp16 autocast
    when the model's weights are half-precision (see load_gpt_checkpoint's
    `dtype`), else a no-op. Autocast -- rather than casting every buffer --
    mirrors the bf16-mixed *training* regime: matmuls run in the low
    precision while the norm/softmax-style ops autocast keeps in fp32 stay
    fp32, and fp32 buffers (e.g. RoPE tables) mix in without dtype errors."""
    p = next(model.parameters(), None)
    if p is not None and p.dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type=torch.device(device).type, dtype=p.dtype)
    return nullcontext()


@dataclass
class SamplingConfig:
    temperature: float = 0.8  # 0 -> greedy decoding
    top_k: int | None = 50  # None -> disabled
    top_p: float | None = None  # nucleus sampling; None -> disabled
    max_new_tokens: int = 256
    # Anti-repetition shaping -- the practical fix for small models looping on
    # a phrase. repetition_penalty is the HF/CTRL multiplicative form (divide a
    # seen token's positive logit / multiply a negative one by the factor);
    # frequency/presence are OpenAI's additive forms (per-occurrence /
    # once-if-present logit subtraction). All default OFF: GRPO rollouts must
    # sample the policy's own distribution -- a shaped sampling distribution
    # would no longer match the teacher-forced log-probs the importance ratio
    # is computed from -- so penalties are enabled per-CLI (chat/api), never
    # baked into the dataclass defaults.
    repetition_penalty: float = 1.0  # 1.0 -> disabled; typical 1.05-1.3
    frequency_penalty: float = 0.0  # 0 -> disabled; typical 0-1
    presence_penalty: float = 0.0  # 0 -> disabled; typical 0-1

    def penalized(self) -> bool:
        """Whether any anti-repetition term is active (callers then need to
        thread the context history into sample())."""
        return (
            self.repetition_penalty != 1.0
            or self.frequency_penalty != 0.0
            or self.presence_penalty != 0.0
        )

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
            elif key == "repetition_penalty":
                value = 1.0 if raw.lower() in ("none", "off") else float(raw)
                if value <= 0:
                    raise ValueError
                self.repetition_penalty = value
            elif key == "frequency_penalty":
                self.frequency_penalty = (
                    0.0 if raw.lower() in ("none", "off") else float(raw)
                )
            elif key == "presence_penalty":
                self.presence_penalty = (
                    0.0 if raw.lower() in ("none", "off") else float(raw)
                )
            else:
                raise ValueError(
                    f"unknown setting '{key}' (temperature, top_k, top_p, "
                    "max_new_tokens, repetition_penalty, frequency_penalty, "
                    "presence_penalty)"
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
            f"repetition_penalty="
            f"{self.repetition_penalty:g}  "
            f"max_new_tokens={self.max_new_tokens}"
        )


def chat_stop_ids(tokenizer: Encoding) -> set[int]:
    """The token ids that end a generated turn: <|im_end|> (the ChatML stop
    token) and <|end_of_text|>. Shared by generate/generate_speculative and
    GRPO rollouts."""
    return {
        tokenizer.encode_single_token(IM_END),
        tokenizer.encode_single_token(EOS_TOKEN),
    }


def _apply_repetition_penalties(
    logits: Tensor, cfg: SamplingConfig, history: Tensor
) -> Tensor:
    """Shape `logits` (B, V) against the tokens already in `history` (B, T):
    the HF/CTRL multiplicative repetition penalty plus OpenAI's additive
    frequency (per occurrence) and presence (once if present) penalties.
    History covers the whole context (prompt + generated), so echoing the
    prompt is discouraged too."""
    counts = torch.zeros_like(logits).scatter_add_(
        -1, history, torch.ones_like(history, dtype=logits.dtype)
    )
    seen = counts > 0
    if cfg.repetition_penalty != 1.0:
        seen_logits = logits[seen]
        logits = logits.masked_scatter(
            seen,
            torch.where(
                seen_logits > 0,
                seen_logits / cfg.repetition_penalty,
                seen_logits * cfg.repetition_penalty,
            ),
        )
    if cfg.frequency_penalty != 0.0:
        logits = logits - cfg.frequency_penalty * counts
    if cfg.presence_penalty != 0.0:
        logits = logits - cfg.presence_penalty * seen.to(logits.dtype)
    return logits


def sample(
    logits: Tensor, cfg: SamplingConfig, history: Tensor | None = None
) -> Tensor:
    """logits: (B, V) -> next token ids (B, 1). Anti-repetition shaping (when
    `history`, the (B, T) context token ids, is provided and cfg enables it),
    then top-k and top-p filtering, both optional; temperature <= 0
    short-circuits to greedy -- after the penalties, which deliberately steer
    greedy decoding away from loops too."""
    if history is not None and history.numel() > 0 and cfg.penalized():
        logits = _apply_repetition_penalties(logits.clone(), cfg, history)
    if cfg.temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / cfg.temperature
    if cfg.top_k is not None and cfg.top_k < logits.shape[-1]:
        threshold = torch.topk(logits, cfg.top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if cfg.top_p is not None and cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # Drop tokens once the mass *before* them already reaches top_p, so
        # the most probable token always survives.
        drop = (cum - probs) >= cfg.top_p
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
    prompt_embeds: Tensor | None = None,
) -> Iterator[int]:
    """Stream token ids continuing `prompt_ids` (KV-cached decode) until
    `<|im_end|>` (the ChatML stop token), `<|end_of_text|>`, or the token
    budget. `model` is a TransformerLM (e.g. gpt.model). Lazy: breaking out
    of the loop aborts generation immediately.

    `max_seq_len` is the model's positional range (its RoPE tables): decoding
    past it asserts, so the budget is capped to stop generation early instead.
    The caller is responsible for a prompt that already fits (see
    ChatApp._build_prompt in scripts/chat.py).

    `prompt_embeds` (1, len(prompt_ids), d_model) prefills from pre-spliced
    embeddings instead of the token ids -- the multimodal path, where media
    soft tokens sit at the placeholder positions (see
    picochat.model.multimodal.splice_media_embeds); `prompt_ids` still sets
    the prompt length for the budget math. Generation then continues on
    ordinary token ids.

    Greedy decoding (temperature <= 0) on a model with MTP heads routes through
    generate_speculative: the emitted stream is identical (every draft is
    verified against the model's own next-token argmax), it just arrives in
    fewer forwards -- so every caller gets the speedup for free. (Embeds
    prefills skip that routing: the speculative path re-commits prompt tokens
    by id on rollback, which would lose the spliced media.)"""
    if (
        cfg.temperature <= 0
        and getattr(model, "n_mtp", 0) > 0
        and prompt_embeds is None
        and not cfg.penalized()
        # speculative drafting verifies against the *unshaped* argmax, so a
        # penalized greedy stream must take the plain path to stay faithful
    ):
        yield from generate_speculative(
            model, tokenizer, prompt_ids, cfg, device, max_seq_len
        )
        return

    budget = cfg.max_new_tokens
    if max_seq_len is not None:
        budget = min(budget, max_seq_len - len(prompt_ids))
        if budget <= 0:
            return

    stop_ids = chat_stop_ids(tokenizer)

    ctx = inference_autocast(model, device)
    if prompt_embeds is not None:
        with ctx:
            logits, cache, pos = model.decode(inputs_embeds=prompt_embeds.to(device))
    else:
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with ctx:
            logits, cache, pos = model.decode(x)
    # The anti-repetition penalties need the running context; only carry it
    # when they are on (GRPO and greedy eval paths keep zero overhead).
    history = None
    if cfg.penalized():
        history = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    next_token = sample(logits[:, -1].float(), cfg, history)

    for _ in range(budget):
        token_id = int(next_token.item())
        if token_id in stop_ids:
            return
        yield token_id
        with ctx:
            logits, cache, pos = model.decode(next_token, cache, pos)
        if history is not None:
            history = torch.cat([history, next_token], dim=1)
        next_token = sample(logits[:, -1].float(), cfg, history)


def _snapshot_cache(cache: list | None) -> list | None:
    """A cheap rollback point for speculative decoding: a shallow copy of the
    per-layer cache list. The mixers' decode() never mutates cached tensors in
    place -- a Gated DeltaNet returns a fresh (recurrent_state, conv_state) and a
    Native Sparse Attention returns a new raw-K/V dict (torch.cat makes new
    tensors) -- so preserving the old element references is enough to restore the
    pre-draft state, without cloning any tensor."""
    return None if cache is None else list(cache)


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

    Cache rollback works for the hybrid GDN/NSA stack by snapshotting the cache
    before each verify forward (see _snapshot_cache) and, only when a draft is
    rejected, restoring it and re-committing the accepted prefix in one extra
    forward -- the recurrent GDN state cannot be sliced back like a KV cache, so
    it is re-derived. When every draft is accepted no recommit is needed, so the
    fast path stays one forward per step.

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
    stop_ids = chat_stop_ids(tokenizer)

    ctx = inference_autocast(model, device)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with ctx:
        logits, mtp, cache, pos = model.decode_heads(x)
    cand = [int(logits[0, -1].argmax())] + [int(m[0, -1].argmax()) for m in mtp]

    emitted = 0
    while emitted < budget:
        if max_seq_len is not None:  # keep the verify chunk within the RoPE range
            cand = cand[: max_seq_len - pos]
            if not cand:
                return
        ct = torch.tensor([cand], dtype=torch.long, device=device)
        base, base_pos = _snapshot_cache(cache), pos
        with ctx:
            logits, mtp, cache, pos = model.decode_heads(ct, cache, pos)
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
        if (len(cand) - 1) - accepted > 0:
            # some drafts rejected: restore the pre-verify cache and re-commit
            # exactly the accepted prefix so the recurrent state ends up at the
            # last accepted token (the correction true[accepted] and the fresh
            # drafts then come from this recommit forward).
            cache, pos = base, base_pos
            commit = torch.tensor(
                [cand[: accepted + 1]], dtype=torch.long, device=device
            )
            with ctx:
                logits, mtp, cache, pos = model.decode_heads(commit, cache, pos)
        # draft afresh from the last committed position (index `accepted` in the
        # chunk that produced the current logits/mtp).
        cand = [int(logits[0, accepted].argmax())] + [
            int(mtp[j][0, accepted].argmax()) for j in range(k)
        ]


class ChatSession:
    """Incremental multi-turn decoding: keeps the model cache (GDN-2 recurrent
    states + NSA KV) across turns and, when a new prompt extends the committed
    conversation, prefills only the delta -- the previous reply's closing
    tokens plus the new user turn -- instead of the whole history. On a long
    conversation that turns per-reply prefill from O(conversation) into
    O(new turn).

    The invariant is `self._ids == the tokens decoded into self._cache`, in
    order. Generation commits each emitted token as it is produced, so an
    aborted stream leaves a consistent (shorter) committed prefix. A prompt
    that does not extend the committed prefix -- history trimmed, /reset, an
    edited turn -- resets the cache and prefills from scratch: the recurrent
    GDN state cannot be rolled back to an arbitrary earlier position.

    Multi-token-prediction speculative decoding is not routed here (its
    rollback protocol conflicts with abort-at-any-yield commitment); plain
    sampled chat -- the TUI/API default -- is unaffected.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Encoding,
        device: torch.device | str = "cpu",
        max_seq_len: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self._ids: list[int] = []
        self._cache = None
        self._pos = 0

    def reset(self) -> None:
        self._ids, self._cache, self._pos = [], None, 0

    @property
    def cached_tokens(self) -> int:
        return len(self._ids)

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], cfg: SamplingConfig) -> Iterator[int]:
        """Stream a reply to `prompt_ids` like engine.generate, reusing the
        committed cache when the prompt extends it."""
        budget = cfg.max_new_tokens
        if self.max_seq_len is not None:
            budget = min(budget, self.max_seq_len - len(prompt_ids))
            if budget <= 0:
                return

        n = len(self._ids)
        if not (0 < n <= len(prompt_ids) and prompt_ids[:n] == self._ids):
            self.reset()
        suffix = prompt_ids[len(self._ids) :]
        if not suffix:
            # Identical prompt (regenerate): the last logits are gone and the
            # recurrent state cannot rewind one token, so start over.
            self.reset()
            suffix = prompt_ids

        ctx = inference_autocast(self.model, self.device)
        stop_ids = chat_stop_ids(self.tokenizer)
        x = torch.tensor([suffix], dtype=torch.long, device=self.device)
        with ctx:
            logits, self._cache, self._pos = self.model.decode(
                x, self._cache, self._pos
            )
        self._ids.extend(suffix)

        history = None
        if cfg.penalized():
            history = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        next_token = sample(logits[:, -1].float(), cfg, history)

        for _ in range(budget):
            token_id = int(next_token.item())
            if token_id in stop_ids:
                return
            yield token_id
            with ctx:
                logits, self._cache, self._pos = self.model.decode(
                    next_token, self._cache, self._pos
                )
            self._ids.append(token_id)
            if history is not None:
                history = torch.cat([history, next_token], dim=1)
            next_token = sample(logits[:, -1].float(), cfg, history)


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


def add_sampling_args(
    parser,
    *,
    temperature: float = 0.8,
    temp_help: str = "0 -> greedy decoding",
    repetition_penalty: float = 1.1,
) -> None:
    """Add the shared --temperature/--top-k/--top-p/--max-new-tokens and
    anti-repetition flags to an argparse parser (used by chat.py, api.py and
    code_eval.py); pair with sampling_from_args. The chat-facing CLIs default
    to a mild repetition penalty (small models loop badly without one);
    benchmark CLIs pass repetition_penalty=1.0 to measure the raw model."""
    parser.add_argument(
        "--temperature", type=float, default=temperature, help=temp_help
    )
    parser.add_argument("--top-k", type=int, default=50, help="0 -> disabled")
    parser.add_argument("--top-p", type=float, default=1.0, help="1.0 -> disabled")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=repetition_penalty,
        help="multiplicative penalty on already-seen tokens; 1.0 -> disabled",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="additive per-occurrence penalty (OpenAI-style); 0 -> disabled",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="additive once-if-present penalty (OpenAI-style); 0 -> disabled",
    )


def sampling_from_args(args) -> SamplingConfig:
    """Build a SamplingConfig from add_sampling_args' parsed flags, applying the
    disabling conventions (top_k 0 -> None, top_p outside (0, 1) -> None)."""
    return SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if 0 < args.top_p < 1 else None,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
    )
