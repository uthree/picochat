"""Generative pass@1 evaluation on verifiable code tasks.

Complements base_eval.py, which scores multiple choice by log-likelihood and so
can't see what GRPO post-training changes: here the model actually *writes*
code, the reply's code is executed against the task's unit tests, and pass@1 is
the fraction of tasks whose tests pass. Run it on the checkpoints before and
after grpo_train.py to measure what the RL stage bought.

Tasks use GRPO's own JSONL format (picochat.rl.grpo.load_tasks, e.g.
configs/grpo/sample_tasks.jsonl):

    {"prompt": "<instruction>", "test": "<python asserting on the answer>"}

Each prompt is rendered as a ChatML user turn (optionally behind --system); the
reply's last fenced code block runs against `test` in the isolation sandbox
(picochat.rl.sandbox, same as GRPO training). Tasks without a `test` field are
skipped -- they would need the LLM judge, and this eval stays verifiable-only.
Decoding is greedy by default: deterministic, and on an MTP checkpoint
generate() routes it through self-speculative decoding automatically.

    python scripts/code_eval.py --checkpoint weights/grpo/last.ckpt \\
        --tasks configs/grpo/sample_tasks.jsonl
"""

import argparse
import json

from picochat.rl import sandbox
from picochat.inference.engine import (
    SamplingConfig,
    add_sampling_args,
    generate,
    resolve_device,
    sampling_from_args,
)
from picochat.rl.grpo import load_tasks
from picochat.rl.reward import extract_code, run_tests_verbose
from picochat.training import load_gpt_checkpoint


def evaluate_tasks(
    model,
    tokenizer,
    samples: list[dict],
    sampling: SamplingConfig,
    device,
    max_seq_len: int,
) -> list[dict]:
    """Generate one reply per tested sample and run its tests. Returns one
    {"prompt", "passed", "response", "output"} record per sample (untested
    samples are the caller's problem: filter before calling)."""
    results = []
    for i, s in enumerate(samples):
        reply_ids = list(
            generate(
                model,
                tokenizer,
                s["prompt_ids"],
                sampling,
                device,
                max_seq_len=max_seq_len,
            )
        )
        response = tokenizer.decode(reply_ids)
        outcome = run_tests_verbose(extract_code(response), s["task"])
        results.append(
            {
                "prompt": s["prompt_str"],
                "passed": outcome.passed,
                "fraction": outcome.fraction,  # per-test-case partial credit
                "response": response,
                "output": outcome.output,
            }
        )
        head = s["prompt_str"].replace("\n", " ")[:60]
        print(
            f"[{i + 1}/{len(samples)}] "
            f"{'PASS' if outcome.passed else f'FAIL {outcome.fraction:.0%}'}  {head}",
            flush=True,
        )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="path to a .ckpt file")
    p.add_argument("--tokenizer", type=str, default="weights/tokenizer.json")
    p.add_argument(
        "--tasks",
        type=str,
        default="configs/grpo/sample_tasks.jsonl",
        help="JSONL task file in GRPO's format ({prompt, test} per line)",
    )
    p.add_argument(
        "--system",
        type=str,
        default=None,
        help="optional system prompt, prepended as a ChatML system turn",
    )
    add_sampling_args(
        p, temperature=0.0, temp_help="0 -> greedy (deterministic; the default)"
    )
    p.add_argument(
        "--limit", type=int, default=None, help="cap the number of tasks (smoke runs)"
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None, help="also write results JSON")
    p.add_argument(
        "--sandbox",
        type=str,
        default="auto",
        help="isolation for the generated code the tests execute: auto|bwrap|none "
        "(see scripts/grpo_train.py)",
    )
    args = p.parse_args()

    # The generated code runs under the sandbox; fail fast if 'bwrap' is
    # required but unavailable, before spending time on generation.
    sandbox.configure(args.sandbox)
    sandbox.check()

    device = resolve_device(args.device)
    print(f"loading model from {args.checkpoint} (device={device}) ...", flush=True)
    gpt, tokenizer = load_gpt_checkpoint(args.checkpoint, args.tokenizer, device)
    max_seq_len = (gpt.hparams["model_config"] or {}).get("max_seq_len", 4096)

    samples = load_tasks(args.tasks, tokenizer, args.system)
    tested = [s for s in samples if s["task"] is not None]
    if args.limit is not None:
        tested = tested[: args.limit]
    skipped = len(samples) - len(tested)
    if skipped:
        print(f"skipping {skipped} task(s) without tests (verifiable-only eval)")
    if not tested:
        raise SystemExit(f"no tested tasks in {args.tasks}")

    results = evaluate_tasks(
        gpt.model, tokenizer, tested, sampling_from_args(args), device, max_seq_len
    )

    passed = sum(r["passed"] for r in results)
    summary = {
        "checkpoint": args.checkpoint,
        "tasks": args.tasks,
        "pass@1": passed / len(results),
        "passed": passed,
        "total": len(results),
        "skipped_untested": skipped,
    }
    print(f"\npass@1: {passed}/{len(results)} = {summary['pass@1']:.1%}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump({**summary, "results": results}, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
