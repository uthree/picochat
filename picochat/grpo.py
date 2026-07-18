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

Multi-step (agentic) RL: with `max_turns > 1`, a tested task runs as an episode
instead of a single response -- the policy proposes code, `agent_rollout` runs
the tests, and on failure feeds the error back as an observation so the policy
can revise, up to the turn budget. The whole trajectory is one member of the
GRPO group and is scored by `picochat.reward.trajectory_reward`, which is tuned
to value eventually reaching a correct answer and staying stable across a long
trial-and-error episode over solving in one shot. Only the policy's own tokens
train (observations are masked out); everything downstream -- advantages, loss,
KL -- is unchanged. `max_turns == 1` is exactly the original single-turn path.
"""

from __future__ import annotations

import asyncio

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from picochat.engine import SamplingConfig, sample
from picochat.gpt import TransformerLM
from picochat.reward import (
    AgentRewardConfig,
    CodeAgentEnv,
    CodeTask,
    RewardModel,
    trajectory_reward,
)
from picochat.tokenizer import EOS_TOKEN, IM_END, render_turn
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


@torch.no_grad()
def _generate_turn(
    model: TransformerLM,
    cache,
    pos,
    first_next: Tensor,
    cfg: SamplingConfig,
    stop_ids: set[int],
    max_new: int,
) -> tuple[list[int], object, object]:
    """Generate one assistant turn (batch 1) continuing from `first_next` (an
    already-sampled but uncommitted token). Commits every token to the KV cache
    and stops at a `stop_ids` token or `max_new`. Returns `(tokens, cache, pos)`
    with `tokens` including the stop token if one was emitted -- the cache then
    holds the whole prefix so the next observation can be appended straight on."""
    toks: list[int] = []
    nxt = first_next
    for _ in range(max_new):
        tok = int(nxt[0, 0])
        logits, cache, pos = model.decode(nxt, cache, pos)  # commit tok
        toks.append(tok)
        if tok in stop_ids:
            break
        nxt = sample(logits[:, -1], cfg)
    return toks, cache, pos


@torch.no_grad()
def agent_rollout(
    model: TransformerLM,
    prompt_ids: list[int],
    cfg: SamplingConfig,
    stop_ids: set[int],
    env,
    decode_text,
    encode_observation,
    max_turns: int,
    per_turn_tokens: int,
    device: torch.device | str = "cpu",
    max_seq_len: int | None = None,
) -> tuple[list[int], list[int], list]:
    """Run one agentic episode for a single prompt: generate a turn, let `env`
    run the tests, and on failure append the feedback observation and generate
    again, up to `max_turns`. Returns `(token_ids, action_mask, steps)`:

    - `token_ids`: the full episode sequence (prompt + every generated turn +
      every observation), exactly the tokens fed to the model, so recomputing
      log-probs over it teacher-forces the same distributions that were sampled.
    - `action_mask`: 1 on policy-generated tokens, 0 on the prompt and on
      environment observations -- only the policy's own tokens train.
    - `steps`: the per-turn StepResult list the trajectory reward scores.

    `decode_text(ids)->str` turns a turn's tokens into text for the env;
    `encode_observation(feedback)->list[int]` renders the feedback (plus the
    ChatML turn scaffolding) back into tokens. Both are injected so this loop
    stays independent of the tokenizer.
    """
    token_ids = list(prompt_ids)
    action_mask = [0] * len(prompt_ids)
    steps: list = []

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, cache, pos = model.decode(x)
    nxt = sample(logits[:, -1], cfg)

    for turn in range(max_turns):
        room = per_turn_tokens
        if max_seq_len is not None:
            room = min(room, max_seq_len - len(token_ids))
        if room <= 0:
            break

        toks, cache, pos = _generate_turn(
            model, cache, pos, nxt, cfg, stop_ids, room
        )
        token_ids.extend(toks)
        action_mask.extend([1] * len(toks))

        text = decode_text([t for t in toks if t not in stop_ids])
        result = env.step(text)
        steps.append(result)
        if result.passed or turn == max_turns - 1:
            break

        obs = encode_observation(result.feedback)
        if max_seq_len is not None and len(token_ids) + len(obs) >= max_seq_len:
            break  # no room to revise: end the episode here
        token_ids.extend(obs)
        action_mask.extend([0] * len(obs))
        obs_t = torch.tensor([obs], dtype=torch.long, device=device)
        logits, cache, pos = model.decode(obs_t, cache, pos)
        nxt = sample(logits[:, -1], cfg)

    return token_ids, action_mask, steps


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
        max_turns: int = 1,
        feedback_chars: int = 512,
        agent_reward: dict | AgentRewardConfig | None = None,
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
        # Multi-step (agentic) RL: max_turns>1 turns a tested task into an
        # episode (propose -> run tests -> revise). max_turns==1 keeps the
        # original single-turn behaviour untouched.
        self.max_turns = max_turns
        self.feedback_chars = feedback_chars
        self.agent_reward = (
            AgentRewardConfig(**agent_reward)
            if isinstance(agent_reward, dict)
            else (agent_reward or AgentRewardConfig())
        )
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
        """Pad (token_ids, action_mask, advantage) rows into (seqs, resp_mask,
        adv). action_mask is 1 on policy-generated tokens (0 on prompt and, for
        agentic episodes, environment observations); resp_mask is it shifted to
        align with token_logprobs' (B, L-1) output so only those tokens train."""
        max_len = max(len(ids) for ids, _, _ in rows)
        seqs = torch.full((len(rows), max_len), self.pad_idx, dtype=torch.long)
        mask = torch.zeros(len(rows), max_len, dtype=torch.float)
        adv = torch.zeros(len(rows), dtype=torch.float)
        for i, (ids, action_mask, a) in enumerate(rows):
            seqs[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[i, : len(action_mask)] = torch.tensor(action_mask, dtype=torch.float)
            adv[i] = a
        dev = self.device
        # Drop the first column: token_logprobs predicts seqs[:, 1:], so the
        # mask/targets live on positions 1..L-1.
        return seqs.to(dev), mask[:, 1:].to(dev), adv.to(dev)

    def _decode_text(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids) if self.tokenizer else ""

    def _feedback_message(self, feedback: str) -> str:
        fb = feedback.strip()
        if fb:
            return (
                "The tests did not pass. Output:\n"
                f"{fb}\n"
                "Revise the code so the tests pass."
            )
        return "The tests did not pass. Revise the code so the tests pass."

    def _encode_observation(self, feedback: str) -> list[int]:
        """Render a failed-test observation as tokens to splice mid-episode: the
        assistant turn's trailing newline (the model stopped at <|im_end|> and
        never emitted it), then a user turn carrying the feedback, then the next
        assistant header to cue the revision."""
        newline = self.tokenizer.encode_ordinary("\n")
        u_head, u_body, u_tail = render_turn(
            "user", self._feedback_message(feedback), self.tokenizer
        )
        a_head, _, _ = render_turn("assistant", "", self.tokenizer)
        return newline + u_head + u_body + u_tail + a_head

    def _agentic_group(
        self, sample_: dict
    ) -> tuple[list[tuple[list[int], list[int]]], list[float]]:
        """One prompt's group as multi-turn episodes: roll out `group_size`
        agentic trajectories and score each with the trajectory reward."""
        env = CodeAgentEnv(task=sample_["task"], feedback_chars=self.feedback_chars)
        trajectories: list[tuple[list[int], list[int]]] = []
        rewards: list[float] = []
        for _ in range(self.group_size):
            ids, action_mask, steps = agent_rollout(
                self.model,
                sample_["prompt_ids"],
                self.sampling,
                self.stop_ids(),
                env,
                self._decode_text,
                self._encode_observation,
                max_turns=self.max_turns,
                per_turn_tokens=self.sampling.max_new_tokens,
                device=self.device,
                max_seq_len=self._max_seq_len,
            )
            trajectories.append((ids, action_mask))
            rewards.append(trajectory_reward(steps, self.agent_reward))
        return trajectories, rewards

    def _single_turn_group(
        self, sample_: dict
    ) -> tuple[list[tuple[list[int], list[int]]], list[float]]:
        """One prompt's group as single responses, scored by the RewardModel."""
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
        texts = [self._decode_text(r) for r in responses]
        tasks = [sample_.get("task")] * self.group_size
        prompts = [sample_.get("prompt_str", "")] * self.group_size
        rewards = self._score(prompts, texts, tasks)
        trajectories = [
            (prompt_ids + resp, [0] * len(prompt_ids) + [1] * len(resp))
            for resp in responses
        ]
        return trajectories, rewards

    def _rollout_and_reward(
        self, batch: list[dict]
    ) -> tuple[list[tuple[list[int], list[int], float]], dict[str, float]]:
        """Roll out every prompt, reward the rollouts, and turn each group's
        rewards into advantages. A tested task runs as a multi-turn episode when
        max_turns>1; everything else stays single-turn. Returns unified
        (token_ids, action_mask, advantage) rows plus logging stats."""
        self.model.eval()  # no dropout: sampling and the loss forward must agree
        rows: list[tuple[list[int], list[int], float]] = []
        all_rewards: list[float] = []
        for sample_ in batch:
            task = sample_.get("task")
            if self.max_turns > 1 and task is not None and task.test_code:
                trajectories, rewards = self._agentic_group(sample_)
            else:
                trajectories, rewards = self._single_turn_group(sample_)
            adv = group_advantages(torch.tensor(rewards, dtype=torch.float))
            for (ids, action_mask), a in zip(trajectories, adv.tolist()):
                rows.append((ids, action_mask, a))
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
