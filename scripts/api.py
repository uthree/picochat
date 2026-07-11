"""OpenAI-compatible Chat Completions API server for a trained picochat
checkpoint (see picochat/api.py for the endpoint implementations).

    python scripts/api.py --checkpoint weights/sft-stage1/last.ckpt --port 8000

Implements the subset of the OpenAI API that OpenAI-compatible clients need:
    GET  /v1/models
    POST /v1/chat/completions   (stream: true or false)

For OpenCode, add a custom `@ai-sdk/openai-compatible` provider pointing
options.baseURL at this server's `/v1` and use --model-id as the model key.

Like scripts/chat.py, the architecture is rebuilt from the checkpoint's own
model_config, and generation streams via picochat.generate.generate(); a base
(pretrain-only) checkpoint has never seen ChatML turns, so this is primarily
for SFT checkpoints.
"""

import argparse
from pathlib import Path

import torch
import uvicorn

from picochat.api import create_app
from picochat.generate import SamplingConfig
from picochat.model.gpt import load_gpt_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="path to a .ckpt file")
    p.add_argument("--tokenizer", type=str, default="weights/tokenizer.json")
    p.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="id returned by GET /v1/models (default: the checkpoint's parent dir name)",
    )
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="default; a request may override it",
    )
    p.add_argument("--top-k", type=int, default=50, help="0 -> disabled")
    p.add_argument("--top-p", type=float, default=1.0, help="1.0 -> disabled")
    p.add_argument("--max-new-tokens", type=int, default=256)
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
    gpt, tokenizer = load_gpt_checkpoint(args.checkpoint, args.tokenizer, device)

    default_sampling = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if 0 < args.top_p < 1 else None,
        max_new_tokens=args.max_new_tokens,
    )

    app = create_app(
        gpt.model,
        tokenizer,
        device=device,
        max_seq_len=(gpt.hparams["model_config"] or {}).get("max_seq_len", 4096),
        model_id=args.model_id or Path(args.checkpoint).parent.name,
        default_sampling=default_sampling,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
