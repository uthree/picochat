"""Bootstrap DPO preference pairs by sampling the policy and letting the LLM
judge pick winners.

    python scripts/dpo_setup.py --checkpoint weights/sft-stage1/last.ckpt \\
        --prompts configs/eval/chat_prompts.jsonl --output data/dpo/pairs.jsonl

For every prompt, sample two replies at a diversity-encouraging temperature,
score both with the judge (picochat.rl.reward.HTTPJudge, or --judge mock for
offline smoke runs), and keep the pair as (chosen, rejected) when the score
gap clears --min-gap -- near-ties teach nothing and add label noise. The
output JSONL is exactly what picochat.training.dpo.PreferenceDataset reads:

    {"prompt": [...], "chosen": "...", "rejected": "...",
     "chosen_score": .., "rejected_score": ..}

Prompt input: JSONL with either {"prompt": str} (chat_prompts.jsonl style,
extra keys ignored) or {"messages": [...]} (a multi-turn prefix whose last
turn is the user's).
"""

import argparse
import asyncio
import json
from pathlib import Path

from picochat.inference.engine import (
    add_sampling_args,
    generate,
    resolve_device,
    sampling_from_args,
)
from picochat.rl.reward import HTTPJudge, MockJudge
from picochat.tokenizer import render_chat_prompt
from picochat.training import load_gpt_checkpoint


def read_prompts(path: str) -> list[list[dict]]:
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "messages" in rec:
                prompts.append(rec["messages"])
            else:
                prompts.append([{"role": "user", "content": rec["prompt"]}])
    return prompts


def sample_reply(model, tokenizer, messages, sampling, device, max_seq_len) -> str:
    prompt_ids = render_chat_prompt(messages, tokenizer)
    ids = list(generate(model, tokenizer, prompt_ids, sampling, device, max_seq_len))
    return tokenizer.decode(ids)


async def judge_pairs(judge, records: list[dict], concurrency: int) -> list[float]:
    sem = asyncio.Semaphore(concurrency)

    async def one(prompt: str, response: str) -> float:
        async with sem:
            return await judge.score(prompt, response)

    return await asyncio.gather(
        *(one(r["prompt_text"], r["response"]) for r in records)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="weights/tokenizer.json")
    p.add_argument("--prompts", type=str, default="configs/eval/chat_prompts.jsonl")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--judge", choices=("http", "mock"), default="http")
    p.add_argument("--judge-base-url", type=str, default="http://localhost:8001/v1")
    p.add_argument("--judge-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument(
        "--min-gap",
        type=float,
        default=0.15,
        help="minimum judge-score gap to keep a pair (near-ties are noise)",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=("auto", "bf16", "fp16", "fp32", "int8"),
    )
    # Sampling defaults tuned for pair diversity: high-ish temperature, no
    # repetition penalty (pairs should reflect the raw policy DPO will tune).
    add_sampling_args(
        p,
        temperature=1.0,
        temp_help="sampling for BOTH replies",
        repetition_penalty=1.0,
    )
    args = p.parse_args()

    device = resolve_device(args.device)
    gpt, tokenizer = load_gpt_checkpoint(
        args.checkpoint, args.tokenizer, device, dtype=args.dtype
    )
    max_seq_len = (gpt.hparams["model_config"] or {}).get("max_seq_len", 4096)
    sampling = sampling_from_args(args)

    prompts = read_prompts(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    judge = (
        MockJudge()
        if args.judge == "mock"
        else HTTPJudge(base_url=args.judge_base_url, model=args.judge_model)
    )

    # Two samples per prompt, judged concurrently afterwards (generation is
    # the sequential part; judging overlaps across the whole set).
    records = []
    for i, messages in enumerate(prompts):
        prompt_text = messages[-1]["content"]
        for _ in range(2):
            response = sample_reply(
                gpt.model, tokenizer, messages, sampling, device, max_seq_len
            )
            records.append(
                {"messages": messages, "prompt_text": prompt_text, "response": response}
            )
        print(f"[{i + 1}/{len(prompts)}] sampled", flush=True)

    scores = asyncio.run(judge_pairs(judge, records, args.concurrency))

    kept = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i in range(0, len(records), 2):
            a, b = records[i], records[i + 1]
            sa, sb = scores[i], scores[i + 1]
            if abs(sa - sb) < args.min_gap:
                continue
            chosen, rejected = (a, b) if sa >= sb else (b, a)
            f.write(
                json.dumps(
                    {
                        "prompt": chosen["messages"],
                        "chosen": chosen["response"],
                        "rejected": rejected["response"],
                        "chosen_score": max(sa, sb),
                        "rejected_score": min(sa, sb),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
    print(f"kept {kept}/{len(prompts)} pairs (min gap {args.min_gap}) -> {out_path}")


if __name__ == "__main__":
    main()
