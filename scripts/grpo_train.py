"""GRPO post-training from a YAML recipe (see picochat.grpo, picochat.reward).

    python scripts/grpo_train.py --config configs/grpo/smoke.yml

Like sft_train.py this always continues from an existing checkpoint: the policy
AND the frozen reference are both built from `init_from`'s own `model_config`,
so they start identical and KL(policy || reference) begins at 0. Tasks come
from a JSONL file (one object per line):

    {"prompt": "<instruction>", "test": "<python asserting on the answer>"}
    {"prompt": "<instruction>"}                 # no test -> judged by the LLM

Each prompt is rendered as a single ChatML user turn; the model's reply is
rewarded by picochat.reward (test pass/fail backbone + external judge for the
untested ones). The judge is any OpenAI-compatible endpoint -- `vllm serve ...`
in production, or the deterministic MockJudge for single-GPU verification.
"""

import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from picochat import sandbox
from picochat.config import load_config
from picochat.grpo import GRPOModule, grpo_collate, load_tasks
from picochat.reward import HTTPJudge, MockJudge, RewardModel, RewardConfig
from picochat.trainer import load_lm_from_checkpoint
from picochat.tokenizer import PAD_TOKEN, load_tokenizer


def build_judge(cfg: dict):
    """Judge backend from config: 'mock' (deterministic, for verification) or
    'http' (any OpenAI-compatible endpoint, e.g. a vLLM server)."""
    jcfg = dict(cfg.get("judge", {}))
    backend = jcfg.pop("backend", "mock")
    if backend == "mock":
        return MockJudge(**jcfg)
    if backend == "http":
        return HTTPJudge(**jcfg)
    raise SystemExit(f"unknown judge backend '{backend}' (choices: mock, http)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="GRPO recipe (YAML)")
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto")
    args = p.parse_args()

    cfg = load_config(args.config)

    # Same seed on every rank. GRPO's DataLoader is a plain shuffled list of
    # prompts, so Lightning's default DistributedSampler injection shards it
    # correctly under DDP; rollout sampling then differs per rank because the
    # prompts do.
    L.seed_everything(cfg.get("seed", 42), workers=True)

    # Untrusted policy-generated code runs under the sandbox; resolve the policy
    # and fail fast if 'bwrap' is required but unavailable (before any training).
    sandbox.configure(cfg.get("sandbox", "auto"))
    sandbox.check()

    tokenizer = load_tokenizer(cfg.get("tokenizer", "weights/tokenizer.json"))
    pad_idx = tokenizer.encode_single_token(PAD_TOKEN)
    init_from = cfg["init_from"]
    output_dir = cfg.get("output_dir", "weights/grpo")

    # Rebuild policy + reference from the checkpoint's own model_config so they
    # start identical (KL begins at 0); load the file once for both.
    ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
    policy, model_config = load_lm_from_checkpoint(init_from, tokenizer.n_vocab, ckpt=ckpt)
    reference, _ = load_lm_from_checkpoint(init_from, tokenizer.n_vocab, ckpt=ckpt)
    model_config = {**model_config, "vocab_size": tokenizer.n_vocab}

    reward_cfg = cfg.get("reward", {})
    reward_model = RewardModel(
        judge=build_judge(cfg),
        cfg=RewardConfig(
            w_task=reward_cfg.get("w_task", 1.0),
            w_format=reward_cfg.get("w_format", 0.1),
            judge_when_tested=reward_cfg.get("judge_when_tested", False),
        ),
    )

    optim_cfg = cfg.get("optim", {})
    trainer_cfg = cfg.get("trainer", {})
    grpo_cfg = cfg.get("grpo", {})

    module = GRPOModule(
        policy,
        reference,
        reward_model,
        pad_idx=pad_idx,
        tokenizer=tokenizer,
        group_size=grpo_cfg.get("group_size", 8),
        temperature=grpo_cfg.get("temperature", 1.0),
        top_k=grpo_cfg.get("top_k"),
        top_p=grpo_cfg.get("top_p"),
        max_new_tokens=grpo_cfg.get("max_new_tokens", 256),
        clip_eps=grpo_cfg.get("clip_eps", 0.2),
        kl_coef=grpo_cfg.get("kl_coef", 0.04),
        reward_concurrency=grpo_cfg.get("reward_concurrency", 32),
        max_turns=grpo_cfg.get("max_turns", 1),
        feedback_chars=grpo_cfg.get("feedback_chars", 512),
        agent_reward=cfg.get("agent_reward"),
        lr=optim_cfg.get("lr", 1e-6),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
        optimizer=optim_cfg.get("optimizer", "adamw"),
        muon_lr=optim_cfg.get("muon_lr", 0.002),
        warmup_steps=optim_cfg.get("warmup_steps", 10),
        max_steps=optim_cfg.get("max_steps", 200),
        grad_clip=trainer_cfg.get("grad_clip", 1.0),
        accumulate=trainer_cfg.get("accumulate", 1),
        model_config=model_config,
    )

    samples = load_tasks(cfg["tasks"], tokenizer, cfg.get("system"))
    loader = DataLoader(
        samples,
        batch_size=trainer_cfg.get("batch_size", 1),
        shuffle=True,
        collate_fn=grpo_collate,
    )

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_steps=optim_cfg.get("max_steps", 200),
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        callbacks=[ModelCheckpoint(dirpath=output_dir, save_last=True)],
    )
    trainer.fit(module, loader)


if __name__ == "__main__":
    main()
