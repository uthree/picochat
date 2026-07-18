"""GRPO (Group Relative Policy Optimization) post-training.

The RL stage for coding-agent ability (see picochat.reward for the reward
side). One step:

  1. For each prompt, sample a *group* of `group_size` rollouts from the
     current policy (KV-cached sampling, `rollout`).
  2. Score every rollout (picochat.reward.RewardModel): test pass/fail as the
     backbone, an external LLM judge where tests can't reach.
  3. Turn the group's rewards into advantages by normalizing within the group
     (`group_advantages`) -- GRPO's trick for dropping the value model: the
     group mean is the baseline.
  4. Update the policy with a PPO-style clipped surrogate plus a KL penalty to
     a frozen reference model (`grpo_loss`).

This is deliberately the simplest correct GRPO: one gradient step per batch of
rollouts (on-policy, no multi-epoch reuse), so the importance ratio is 1 at the
update and the objective reduces to REINFORCE-with-group-baseline + KL. The
clipped-ratio machinery is still here so adding inner epochs later is a config
change, not a rewrite. Optimizer / LR-schedule / gradient-accumulation
scaffolding is shared with the other trainers via LMTrainerMixin.
"""

from __future__ import annotations

import asyncio

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from picochat.engine import SamplingConfig, sample
from picochat.gpt import TransformerLM
from picochat.reward import CodeTask, RewardModel
from picochat.tokenizer import EOS_TOKEN, IM_END
from picochat.trainer import LMTrainerMixin


@torch.no_grad()
def rollout(
    model: TransformerLM,
    prompt_ids: list[int],
    group_size: int,
    cfg: SamplingConfig,
    stop_ids: set[int],
    device: torch.device | str = "cpu",
    max_seq_len: int | None = None,
) -> list[list[int]]:
    """Sample `group_size` continuations of one prompt in parallel (one KV cache
    of batch `group_size`), each stopping at a `stop_ids` token or the token
    budget. Returns the response token ids per rollout, excluding the stop
    token. All rows share the same prompt, so no padding is needed here.
    """
    budget = cfg.max_new_tokens
    if max_seq_len is not None:
        budget = min(budget, max_seq_len - len(prompt_ids))
    if budget <= 0:
        return [[] for _ in range(group_size)]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device).repeat(
        group_size, 1
    )
    logits, cache, pos = model.decode(x)
    next_token = sample(logits[:, -1], cfg)  # (G, 1)

    responses: list[list[int]] = [[] for _ in range(group_size)]
    done = [False] * group_size
    for _ in range(budget):
        for i in range(group_size):
            if done[i]:
                continue
            tok = int(next_token[i, 0])
            if tok in stop_ids:
                done[i] = True
            else:
                responses[i].append(tok)
        if all(done):
            break
        logits, cache, pos = model.decode(next_token, cache, pos)
        next_token = sample(logits[:, -1], cfg)
    return responses


def group_advantages(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    """Group-relative advantage: standardize a group's rewards to zero mean,
    unit variance. A group where every rollout scored the same (std 0) carries
    no learning signal, so its advantages are all zero."""
    std = rewards.std()
    if std < eps:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + eps)


def token_logprobs(model: TransformerLM, seqs: Tensor) -> Tensor:
    """Per-token log-prob of the realized next token: position t scores
    seqs[:, t+1] under the model's distribution at t. Returns (B, L-1)."""
    logits = model(seqs)  # (B, L, V)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = seqs[:, 1:].unsqueeze(-1)
    return logp.gather(-1, targets).squeeze(-1)


def grpo_loss(
    policy_lp: Tensor,
    ref_lp: Tensor,
    old_lp: Tensor,
    advantages: Tensor,
    resp_mask: Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
) -> tuple[Tensor, dict[str, float]]:
    """Token-level GRPO objective, averaged over response tokens.

    - PPO clipped surrogate on the importance ratio exp(policy_lp - old_lp),
      weighted by the per-token advantage (the rollout's group advantage,
      broadcast to its tokens).
    - KL(policy || ref) via the k3 estimator (unbiased, non-negative), pulling
      the policy back toward the reference so it doesn't reward-hack away from
      a sensible language model.

    All tensors are (B, L-1) except `advantages` (B,), broadcast over tokens.
    `resp_mask` (B, L-1) is 1 on generated tokens, 0 on prompt/padding.
    """
    adv = advantages.unsqueeze(-1)  # (B, 1) -> broadcast over tokens
    ratio = torch.exp(policy_lp - old_lp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    pg = -torch.min(unclipped, clipped)

    diff = ref_lp - policy_lp
    kl = torch.exp(diff) - diff - 1.0  # k3 estimator, >= 0

    per_token = pg + kl_coef * kl
    denom = resp_mask.sum().clamp(min=1.0)
    loss = (per_token * resp_mask).sum() / denom
    with torch.no_grad():
        metrics = {
            "kl": float((kl * resp_mask).sum() / denom),
            "ratio": float((ratio * resp_mask).sum() / denom),
        }
    return loss, metrics


class GRPOModule(LMTrainerMixin, L.LightningModule):
    """GRPO LightningModule. `self.model` is the trainable policy; a frozen
    `reference` (kept out of the state_dict / optimizer, like GPT's compiled
    `_forward`) supplies the KL anchor. Each training step rolls out, rewards,
    computes advantages and takes one clipped-surrogate + KL step.
    """

    def __init__(
        self,
        transformer_lm: TransformerLM,
        reference_lm: TransformerLM,
        reward_model: RewardModel,
        pad_idx: int,
        tokenizer=None,
        group_size: int = 8,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        max_new_tokens: int = 256,
        clip_eps: float = 0.2,
        kl_coef: float = 0.04,
        reward_concurrency: int = 32,
        lr: float = 1e-6,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        optimizer: str = "adamw",
        muon_lr: float = 0.002,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.01,
        warmup_steps: int = 10,
        max_steps: int | None = None,
        min_lr_ratio: float = 1.0,
        grad_clip: float | None = 1.0,
        accumulate: int = 1,
        model_config: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters("model_config")
        self.model = transformer_lm
        # Frozen reference: stored via object.__setattr__ so nn.Module doesn't
        # register it as a submodule (it would double the checkpoint and feed
        # its params to the optimizer). Moved to device in on_fit_start.
        reference_lm.requires_grad_(False).eval()
        object.__setattr__(self, "reference", reference_lm)
        self.reward_model = reward_model
        self.pad_idx = pad_idx
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.reward_concurrency = reward_concurrency
        self.sampling = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        self._max_seq_len = model_config.get("max_seq_len") if model_config else None
        self.automatic_optimization = False
        self._init_trainer(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            optimizer=optimizer,
            muon_lr=muon_lr,
            muon_momentum=muon_momentum,
            muon_weight_decay=muon_weight_decay,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
            grad_clip=grad_clip,
            accumulate=accumulate,
            tokenizer=tokenizer,
            fused_loss=False,
        )
        self._stop_ids: set[int] | None = None

    def on_fit_start(self) -> None:
        self.reference.to(self.device)

    def stop_ids(self) -> set[int]:
        if self._stop_ids is None:
            if self.tokenizer is None:
                self._stop_ids = set()
            else:
                self._stop_ids = {
                    self.tokenizer.encode_single_token(IM_END),
                    self.tokenizer.encode_single_token(EOS_TOKEN),
                }
        return self._stop_ids

    def _score(
        self, prompts: list[str], responses: list[str], tasks: list[CodeTask | None]
    ) -> list[float]:
        """Reward every rollout (async under the hood; run to completion here so
        the training step stays synchronous)."""
        return asyncio.run(
            self.reward_model.score_group(
                prompts, responses, tasks, concurrency=self.reward_concurrency
            )
        )

    def _build_batch(
        self, rows: list[tuple[list[int], list[int], float]]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Pad (prompt_ids, response_ids, advantage) rows into (seqs, resp_mask,
        adv). resp_mask marks the generated tokens (positions predicted by the
        loss), aligned to token_logprobs' (B, L-1) output."""
        max_len = max(len(p) + len(r) for p, r, _ in rows)
        seqs = torch.full((len(rows), max_len), self.pad_idx, dtype=torch.long)
        mask = torch.zeros(len(rows), max_len, dtype=torch.float)
        adv = torch.zeros(len(rows), dtype=torch.float)
        for i, (prompt, resp, a) in enumerate(rows):
            ids = prompt + resp
            seqs[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            # Response tokens occupy [len(prompt), len(prompt)+len(resp)); mark
            # them so only generated tokens contribute to the loss.
            mask[i, len(prompt) : len(ids)] = 1.0
            adv[i] = a
        dev = self.device
        # Drop the first column: token_logprobs predicts seqs[:, 1:], so the
        # mask/targets live on positions 1..L-1.
        return seqs.to(dev), mask[:, 1:].to(dev), adv.to(dev)

    def _rollout_and_reward(
        self, batch: list[dict]
    ) -> tuple[list[tuple[list[int], list[int], float]], dict[str, float]]:
        """Roll out every prompt, reward the rollouts, and turn each group's
        rewards into advantages. Returns per-rollout (prompt, response,
        advantage) rows plus logging stats."""
        self.model.eval()  # no dropout: sampling and the loss forward must agree
        rows: list[tuple[list[int], list[int], float]] = []
        all_rewards: list[float] = []
        for sample_ in batch:
            prompt_ids = sample_["prompt_ids"]
            responses = rollout(
                self.model,
                prompt_ids,
                self.group_size,
                self.sampling,
                self.stop_ids(),
                device=self.device,
                max_seq_len=self._max_seq_len,
            )
            texts = [
                self.tokenizer.decode(r) if self.tokenizer else "" for r in responses
            ]
            tasks = [sample_.get("task")] * self.group_size
            prompts = [sample_.get("prompt_str", "")] * self.group_size
            rewards = self._score(prompts, texts, tasks)
            adv = group_advantages(torch.tensor(rewards, dtype=torch.float))
            for resp, a in zip(responses, adv.tolist()):
                rows.append((prompt_ids, resp, a))
            all_rewards.extend(rewards)
        stats = {
            "reward_mean": sum(all_rewards) / max(1, len(all_rewards)),
            "reward_max": max(all_rewards) if all_rewards else 0.0,
        }
        return rows, stats

    def training_step(self, batch: list[dict], batch_idx: int) -> Tensor:
        rows, stats = self._rollout_and_reward(batch)
        seqs, resp_mask, adv = self._build_batch(rows)

        policy_lp = token_logprobs(self.model, seqs)
        with torch.no_grad():
            ref_lp = token_logprobs(self.reference, seqs)
        old_lp = policy_lp.detach()  # on-policy single step: ratio == 1 here

        loss, metrics = grpo_loss(
            policy_lp, ref_lp, old_lp, adv, resp_mask, self.clip_eps, self.kl_coef
        )
        (loss / self.accumulate).backward()
        self._optimizer_step(batch_idx)

        self.log("train_loss", loss)
        self.log("loss", loss, prog_bar=True, logger=False)
        self.log("reward", stats["reward_mean"], prog_bar=True)
        self.log("reward_max", stats["reward_max"])
        self.log("kl", metrics["kl"])
        return loss


def grpo_collate(batch: list[dict]) -> list[dict]:
    """Identity collate: GRPO samples carry a CodeTask and variable-length
    prompt ids, so keep the batch as a plain list instead of stacking tensors."""
    return batch
