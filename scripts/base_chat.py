"""Interactive chat REPL for a trained picochat checkpoint.

Loads a model + tokenizer from a checkpoint, then runs a multi-turn ChatML
conversation: each stdin line becomes a user turn, the full history is
rendered via picochat.data.sft.render_chat_prompt (ending in the
`<|im_start|>assistant\\n` cue) and the reply streams to stdout until the
model emits `<|im_end|>` (or `</s>`/the token budget). Pass --system to
prepend a system turn; --no-history makes every exchange independent.

    python scripts/base_chat.py --checkpoint weights/sft/last.ckpt \\
        --system "You are a helpful assistant."

Requires a checkpoint produced by the current scripts/base_train.py or
sft_train.py, which embeds a `model_config` hyperparameter (see GPT.__init__)
recording the exact build_lm() recipe used to construct the model -- the
architecture is rebuilt from the checkpoint itself. Note that a base
(pretrain-only) checkpoint has never seen ChatML turns, so it will ramble;
this REPL is primarily for SFT checkpoints.
"""

import argparse
import sys
from typing import Iterator

import torch
from tiktoken import Encoding

from picochat.data.sft import IM_END, render_chat_prompt
from picochat.model.gpt import GPT, build_lm
from picochat.tokenizer import load_tokenizer


def load_model(
    checkpoint: str, tokenizer_path: str, device: torch.device
) -> tuple[GPT, Encoding]:
    tokenizer = load_tokenizer(tokenizer_path)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"{checkpoint} doesn't look like a Lightning checkpoint")
    model_config = (ckpt.get("hyper_parameters") or {}).get("model_config")
    if model_config is None:
        raise ValueError(
            f"{checkpoint} has no 'model_config' hyperparameter -- it predates "
            "GPT.__init__ saving it, so its architecture can't be rebuilt. "
            "Retrain to produce a checkpoint with model_config."
        )

    print(f"using model_config from checkpoint: {model_config}", flush=True)
    lm = build_lm(**{**model_config, "vocab_size": tokenizer.n_vocab})

    gpt = GPT(lm, compile=False, tokenizer=tokenizer)
    gpt.load_state_dict(ckpt["state_dict"])
    gpt.eval()
    gpt.to(device)
    return gpt, tokenizer


def _sample(
    logits: torch.Tensor, temperature: float, top_k: int | None
) -> torch.Tensor:
    """logits: (B, V) -> next token ids (B, 1)."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    gpt: GPT,
    tokenizer: Encoding,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    temperature: float,
    top_k: int | None,
) -> Iterator[int]:
    # <|im_end|> ends the assistant turn (the ChatML stop token); </s> ends
    # the whole document -- a well-trained SFT model emits the former, but
    # stop on either.
    stop_ids = {
        tokenizer.encode_single_token(IM_END),
        tokenizer.encode_single_token("</s>"),
    }

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, cache, pos = gpt.model.decode(x)
    next_token = _sample(logits[:, -1], temperature, top_k)

    for _ in range(max_new_tokens):
        token_id = next_token.item()
        if token_id in stop_ids:
            return
        yield token_id
        logits, cache, pos = gpt.model.decode(next_token, cache, pos)
        next_token = _sample(logits[:, -1], temperature, top_k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="path to a .ckpt file")
    p.add_argument("--tokenizer", type=str, default="weights/tokenizer.json")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--temperature", type=float, default=0.8, help="0 -> greedy decoding"
    )
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--system",
        type=str,
        default=None,
        help="optional system prompt, prepended as a ChatML system turn",
    )
    p.add_argument(
        "--no-history",
        action="store_true",
        help="treat every exchange as a fresh single-turn conversation",
    )
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"loading model from {args.checkpoint} (device={device}) ...", flush=True)
    gpt, tokenizer = load_model(args.checkpoint, args.tokenizer, device)
    print("", flush=True)
    with open("assets/logo_ascii.txt") as f:  # show ascii art logo
        print(f.read())
    print("", flush=True)
    print("ready. Ctrl-C or Ctrl-D to exit.", flush=True)

    # Conversation state: the running ChatML history. The whole history is
    # re-prefilled each turn (the models are small; simpler than carrying the
    # KV cache across turns).
    system_messages = (
        [{"role": "system", "content": args.system}] if args.system else []
    )
    messages = list(system_messages)
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text.strip():
            continue
        if args.no_history:
            messages = list(system_messages)
        messages.append({"role": "user", "content": text})
        prompt_ids = render_chat_prompt(messages, tokenizer)
        reply_ids: list[int] = []
        for token_id in generate(
            gpt,
            tokenizer,
            prompt_ids,
            args.max_new_tokens,
            device,
            args.temperature,
            args.top_k,
        ):
            reply_ids.append(token_id)
            # Streamed display is per-token (a multi-byte character split
            # across tokens shows replacement chars until complete)...
            sys.stdout.write(
                tokenizer.decode_single_token_bytes(token_id).decode(
                    "utf-8", errors="replace"
                )
            )
            sys.stdout.flush()
        print()
        # ...but the history entry is decoded from the full id sequence, so
        # the next turn's prompt re-encodes clean text.
        messages.append({"role": "assistant", "content": tokenizer.decode(reply_ids)})


if __name__ == "__main__":
    main()
