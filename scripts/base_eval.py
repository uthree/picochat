"""Evaluate a picochat checkpoint on multiple-choice benchmarks.

    # base (pretrain-only) checkpoint: plain text-continuation scoring
    python scripts/base_eval.py --checkpoint weights/stage2/last.ckpt

    # SFT checkpoint: render each item as a ChatML user turn instead
    python scripts/base_eval.py --checkpoint weights/sft-stage1/last.ckpt --chat

Every answer choice is scored by its completion log-likelihood and the
prediction is the argmax (see picochat/tasks.py for tasks and metrics). The
same items back both renderings, so acc/acc_norm are comparable before and
after SFT. Datasets download from the HF hub on first use; --limit N keeps a
quick smoke run cheap. Like chat.py, the architecture is rebuilt from
the model_config embedded in the checkpoint itself.
"""

import argparse
import contextlib
import json

import torch

from picochat.tasks import TASKS, evaluate_task
from picochat.trainer import load_gpt_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="path to a .ckpt file")
    p.add_argument("--tokenizer", type=str, default="weights/tokenizer.json")
    p.add_argument(
        "--tasks",
        type=str,
        default=",".join(TASKS),
        help=f"comma-separated subset of: {','.join(TASKS)}",
    )
    p.add_argument(
        "--chat",
        action="store_true",
        help="render items as ChatML user turns (for SFT checkpoints)",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="cap examples per task (smoke runs)"
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None, help="also write results JSON")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}. choices: {list(TASKS)}")

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
    max_len = (gpt.hparams["model_config"] or {}).get("max_seq_len", 4096)

    # bf16 autocast on CUDA matches the training precision and roughly halves
    # eval time; elsewhere run in full precision.
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    results = []
    print(f"{'task':<14} {'n':>6} {'acc':>7} {'acc_norm':>9} {'random':>7}", flush=True)
    with autocast:
        for task in tasks:
            r = evaluate_task(
                gpt.model,
                tokenizer,
                task,
                chat=args.chat,
                limit=args.limit,
                batch_size=args.batch_size,
                max_len=max_len,
                device=device,
            )
            results.append(r)
            print(
                f"{r['task']:<14} {r['n']:>6} {r['acc']:>7.4f} "
                f"{r['acc_norm']:>9.4f} {r['random']:>7.4f}",
                flush=True,
            )
    mean_acc = sum(r["acc"] for r in results) / len(results)
    mean_norm = sum(r["acc_norm"] for r in results) / len(results)
    print(f"{'mean':<14} {'':>6} {mean_acc:>7.4f} {mean_norm:>9.4f}", flush=True)

    if args.output:
        payload = {
            "checkpoint": args.checkpoint,
            "chat": args.chat,
            "limit": args.limit,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
