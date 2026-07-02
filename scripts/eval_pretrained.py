"""Interactive REPL for a trained picochat checkpoint.

Loads a model + tokenizer from a checkpoint, then repeatedly reads a line
from stdin, encodes it, appends "<s>", and autoregresses (streaming to
stdout) until "</s>" is generated -- then goes back to waiting for the next
input.

    python scripts/eval_pretrained.py --checkpoint weights/stage3/last.ckpt

Requires a checkpoint produced by the current scripts/pretrain.py, which
embeds a `model_config` hyperparameter (see GPT.__init__) recording the
exact build_lm() recipe used to construct the model -- the architecture is
rebuilt from the checkpoint itself. Checkpoints predating that (no
`model_config` hparam) aren't supported; retrain or patch one in by hand.
"""

import argparse
import sys
from typing import Iterator

import torch
from tiktoken import Encoding

from picochat.model.gpt import GPT, build_lm
from picochat.tokenizer import load_tokenizer


def load_model(checkpoint: str, tokenizer_path: str, device: torch.device) -> tuple[GPT, Encoding]:
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


def _sample(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
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
    """Yield generated token ids, starting decode from "<s>" and stopping at "</s>"."""
    bos = tokenizer._special_tokens["<s>"]
    eos = tokenizer._special_tokens["</s>"]

    x = torch.tensor([prompt_ids + [bos]], dtype=torch.long, device=device)
    logits, cache, pos = gpt.model.decode(x)
    next_token = _sample(logits[:, -1], temperature, top_k)

    for _ in range(max_new_tokens):
        token_id = next_token.item()
        if token_id == eos:
            return
        yield token_id
        logits, cache, pos = gpt.model.decode(next_token, cache, pos)
        next_token = _sample(logits[:, -1], temperature, top_k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="path to a .ckpt file")
    p.add_argument("--tokenizer", type=str, default="weights/tokenizer.json")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8, help="0 -> greedy decoding")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", type=str, default=None)
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
    print("ready. Ctrl-C or Ctrl-D to exit.", flush=True)

    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text.strip():
            continue
        prompt_ids = tokenizer.encode(text, disallowed_special=())
        for token_id in generate(
            gpt,
            tokenizer,
            prompt_ids,
            args.max_new_tokens,
            device,
            args.temperature,
            args.top_k,
        ):
            sys.stdout.write(
                tokenizer.decode_single_token_bytes(token_id).decode(
                    "utf-8", errors="replace"
                )
            )
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    main()
