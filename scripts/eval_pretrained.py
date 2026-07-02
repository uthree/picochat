"""Interactive REPL for a trained picochat checkpoint.

Loads a model + tokenizer from a checkpoint, then repeatedly reads a line
from stdin, encodes it, appends "<s>", and autoregresses (streaming to
stdout) until "</s>" is generated -- then goes back to waiting for the next
input.

    python scripts/eval_pretrained.py --checkpoint weights/stage3/last.ckpt

Checkpoints produced by the current scripts/pretrain.py embed a
`model_config` hyperparameter (see GPT.__init__) recording the exact
build_lm() recipe used to construct the model, so the architecture is
rebuilt from the checkpoint itself. Older checkpoints saved before that was
added don't have it; for those, pass --size/--d-model/... to describe the
architecture by hand (mismatches fail as a state_dict shape/key error).
"""

import argparse
import sys
from typing import Iterator

import torch
from tiktoken import Encoding

from picochat.model.gpt import GPT, MODEL_PRESETS, build_lm
from picochat.tokenizer import load_tokenizer

MODEL_OVERRIDES = (
    "d_model",
    "n_heads",
    "n_kv_heads",
    "n_layers",
    "tie_embeddings",
    "grad_checkpoint",
    "window_size",
)


def load_model(
    checkpoint: str,
    tokenizer_path: str,
    device: torch.device,
    size: str,
    max_seq_len: int,
    overrides: dict,
) -> tuple[GPT, Encoding]:
    tokenizer = load_tokenizer(tokenizer_path)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    is_lightning_ckpt = isinstance(ckpt, dict) and "state_dict" in ckpt
    model_config = (ckpt.get("hyper_parameters") or {}).get("model_config") if is_lightning_ckpt else None

    if model_config is not None:
        print(f"using model_config from checkpoint: {model_config}", flush=True)
        model_config = {**model_config, "vocab_size": tokenizer.n_vocab}
        lm = build_lm(**model_config)
    else:
        print("checkpoint has no model_config hparam; using --size/CLI overrides", flush=True)
        lm = build_lm(
            size, vocab_size=tokenizer.n_vocab, max_seq_len=max_seq_len, **overrides
        )

    gpt = GPT(lm, compile=False, tokenizer=tokenizer)
    state = ckpt["state_dict"] if is_lightning_ckpt else ckpt
    gpt.load_state_dict(state)
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
    p.add_argument("--size", type=str, default="pico", choices=list(MODEL_PRESETS))
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--n-kv-heads", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument(
        "--tie-embeddings", type=lambda s: s.lower() != "false", default=None
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8, help="0 -> greedy decoding")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    overrides = {
        k.replace("-", "_"): getattr(args, k.replace("-", "_"))
        for k in MODEL_OVERRIDES
        if k != "grad_checkpoint" and getattr(args, k.replace("-", "_")) is not None
    }

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"loading model from {args.checkpoint} (device={device}) ...", flush=True)
    gpt, tokenizer = load_model(
        args.checkpoint, args.tokenizer, device, args.size, args.max_seq_len, overrides
    )
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
