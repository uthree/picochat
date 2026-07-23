"""MT-Bench-style judged chat-quality evaluation.

Complements base_eval.py (multiple-choice log-likelihood) and code_eval.py
(verifiable pass@1): open-ended chat quality has no unit tests, so here the
model answers a fixed set of single-turn prompts across the MT-Bench category
mix (writing, roleplay, reasoning, math, coding, extraction, stem, humanities;
half Japanese, half English -- see configs/eval/chat_prompts.jsonl) and an LLM
judge grades every reply against a yes/no chat-quality checklist. Scores are
in [0, 1]; the summary reports the per-category means and the overall mean,
so before/after comparisons of an SFT or RL stage show *where* quality moved,
not just whether it did.

Two judges, same interface (picochat.rl.reward):

- `--judge mock` (default): the deterministic offline MockJudge -- wiring
  verification only, not a real quality measure.
- `--judge http`: HTTPJudge against any OpenAI-compatible endpoint (e.g.
  `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001`), with the chat
  checklist below instead of the code-centric default.

Decoding is sampled (temperature 0.7) with the chat-default repetition
penalty: this eval measures the model as chat users experience it, unlike the
raw-model benchmark CLIs.

    python scripts/chat_eval.py --checkpoint weights/sft-stage1/last.ckpt \\
        --judge http --judge-base-url http://localhost:8001/v1
"""

import argparse
import asyncio
import json

from picochat.inference.engine import (
    SamplingConfig,
    add_sampling_args,
    generate,
    resolve_device,
    sampling_from_args,
)
from picochat.rl.reward import HTTPJudge, JudgeBackend, MockJudge
from picochat.tokenizer import render_chat_prompt
from picochat.training import load_gpt_checkpoint

DEFAULT_PROMPTS = "configs/eval/chat_prompts.jsonl"

# The MT-Bench category set: broad coverage of what a chat assistant is asked
# to do, kept identical so scores are comparable with the wider ecosystem.
CATEGORIES = (
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
)

# Chat-quality checklist for HTTPJudge (replaces its code-centric default):
# each item is a near-binary judgement about one facet of a good chat reply,
# ordered from the outcome that matters most (helpfulness/correctness) to
# style. The same-language item matters for a bilingual prompt set -- a small
# model drifting into the wrong language is a real, common failure the other
# items would miss.
CHAT_QUESTIONS = (
    "Is the response helpful -- does it directly address what the user asked "
    "for, rather than a related or easier question?",
    "Does the response follow all explicit instructions in the prompt "
    "(requested format, length, number of items, constraints)?",
    "Is the content factually and logically correct, with no errors you can identify?",
    "Is the response complete -- no part of the question left unanswered, "
    "half-done, or trailing off mid-sentence?",
    "Is the response written in the same language as the prompt (unless the "
    "prompt explicitly asks for another language)?",
    "Is the response fluent and non-redundant -- natural wording, no garbled "
    "text, and no restating the same content multiple times?",
)

# Helpfulness and correctness dominate; instruction-following and completeness
# matter next; language match and fluency are guardrails that shouldn't be
# able to carry a wrong answer to a good score.
CHAT_WEIGHTS = (3.0, 2.0, 3.0, 2.0, 1.0, 1.0)


def load_prompts(path: str, limit: int | None = None) -> list[dict]:
    """Read the prompt JSONL ({"category", "prompt"} per line; blank lines
    skipped) and optionally cap it. Validation is strict-but-cheap: a typo'd
    category would silently create a new table row, so fail fast instead."""
    prompts = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "category" not in row or "prompt" not in row:
                raise ValueError(f"{path}:{lineno}: needs 'category' and 'prompt'")
            if row["category"] not in CATEGORIES:
                raise ValueError(
                    f"{path}:{lineno}: unknown category {row['category']!r} "
                    f"(choices: {', '.join(CATEGORIES)})"
                )
            prompts.append({"category": row["category"], "prompt": row["prompt"]})
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def evaluate_prompts(
    model,
    tokenizer,
    prompts: list[dict],
    sampling: SamplingConfig,
    judge: JudgeBackend,
    device,
    max_seq_len: int,
    concurrency: int = 8,
    system: str | None = None,
) -> list[dict]:
    """Generate one reply per prompt, then judge every (prompt, reply) pair.
    Returns one {"category", "prompt", "response", "score"} record per prompt.

    Generation is sequential -- one model on one device, so interleaving
    replies buys nothing -- but judging goes over the network, so all pairs
    are scored concurrently under a bounded semaphore (the same pattern as
    RewardModel.score_group). A judge failure scores 0.0 rather than raising
    (HTTPJudge already degrades that way), so one flaky call can't kill an
    hour of generation."""
    responses = []
    for i, p in enumerate(prompts):
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": p["prompt"]})
        prompt_ids = render_chat_prompt(messages, tokenizer)
        reply_ids = list(
            generate(
                model, tokenizer, prompt_ids, sampling, device, max_seq_len=max_seq_len
            )
        )
        responses.append(tokenizer.decode(reply_ids))
        head = p["prompt"].replace("\n", " ")[:60]
        print(f"[{i + 1}/{len(prompts)}] {p['category']:<11} {head}", flush=True)

    async def score_all() -> list[float]:
        sem = asyncio.Semaphore(concurrency)

        async def one(prompt: str, response: str) -> float:
            async with sem:
                return await judge.score(prompt, response)

        return await asyncio.gather(
            *(one(p["prompt"], r) for p, r in zip(prompts, responses))
        )

    scores = asyncio.run(score_all())
    return [
        {
            "category": p["category"],
            "prompt": p["prompt"],
            "response": r,
            "score": float(s),
        }
        for p, r, s in zip(prompts, responses, scores)
    ]


def aggregate(results: list[dict]) -> dict:
    """Per-category mean scores plus the overall mean. The overall mean is over
    *all* results (not the mean of category means), so with the balanced
    built-in set they coincide, and with --limit the number stays honest about
    what was actually run."""
    by_category: dict[str, list[float]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["score"])
    return {
        "overall": sum(r["score"] for r in results) / len(results),
        "categories": {
            c: sum(scores) / len(scores) for c, scores in sorted(by_category.items())
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="path to a .ckpt file")
    p.add_argument("--tokenizer", type=str, default="weights/tokenizer.json")
    p.add_argument(
        "--prompts",
        type=str,
        default=DEFAULT_PROMPTS,
        help="JSONL prompt file ({category, prompt} per line)",
    )
    p.add_argument(
        "--system",
        type=str,
        default=None,
        help="optional system prompt, prepended as a ChatML system turn",
    )
    # temperature 0.7 + the chat-default repetition penalty: measure the model
    # as chat users experience it (the benchmark CLIs measure raw greedy).
    add_sampling_args(p, temperature=0.7)
    p.add_argument(
        "--judge",
        type=str,
        default="mock",
        choices=("mock", "http"),
        help="mock = deterministic offline stand-in; http = real LLM judge",
    )
    p.add_argument(
        "--judge-base-url",
        type=str,
        default="http://localhost:8001/v1",
        help="OpenAI-compatible endpoint for --judge http",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="model name at the judge endpoint for --judge http",
    )
    p.add_argument(
        "--concurrency", type=int, default=8, help="max in-flight judge calls"
    )
    p.add_argument(
        "--limit", type=int, default=None, help="cap the number of prompts (smoke runs)"
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=("auto", "bf16", "fp16", "fp32"),
        help="inference weight dtype; auto = bf16 on CUDA, fp32 elsewhere",
    )
    p.add_argument("--output", type=str, default=None, help="also write results JSON")
    args = p.parse_args()

    prompts = load_prompts(args.prompts, args.limit)
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts}")

    if args.judge == "http":
        judge = HTTPJudge(
            base_url=args.judge_base_url,
            model=args.judge_model,
            questions=CHAT_QUESTIONS,
            weights=CHAT_WEIGHTS,
        )
    else:
        judge = MockJudge()
        print("using MockJudge: scores verify wiring, not chat quality", flush=True)

    device = resolve_device(args.device)
    print(f"loading model from {args.checkpoint} (device={device}) ...", flush=True)
    gpt, tokenizer = load_gpt_checkpoint(
        args.checkpoint, args.tokenizer, device, dtype=args.dtype
    )
    max_seq_len = (gpt.hparams["model_config"] or {}).get("max_seq_len", 4096)

    results = evaluate_prompts(
        gpt.model,
        tokenizer,
        prompts,
        sampling_from_args(args),
        judge,
        device,
        max_seq_len,
        concurrency=args.concurrency,
        system=args.system,
    )
    summary = aggregate(results)

    print(f"\n{'category':<12} {'n':>4} {'score':>7}", flush=True)
    for cat, mean in summary["categories"].items():
        n = sum(r["category"] == cat for r in results)
        print(f"{cat:<12} {n:>4} {mean:>7.4f}", flush=True)
    print(f"{'overall':<12} {len(results):>4} {summary['overall']:>7.4f}", flush=True)

    if args.output:
        payload = {
            "checkpoint": args.checkpoint,
            "prompts": args.prompts,
            "judge": args.judge,
            "overall": summary["overall"],
            "categories": summary["categories"],
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
