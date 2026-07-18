"""Verify the GRPO pieces on a tiny CPU model: the pure reward->advantage->loss
math, batched rollout with stop tokens, and a few real optimizer steps through
a Lightning Trainer (params move, nothing goes NaN)."""

import copy

import lightning as L
import torch
from torch.utils.data import DataLoader

from picochat import grpo
from picochat.engine import SamplingConfig
from picochat.gpt import TransformerLM
from picochat.reward import MockJudge, RewardModel
from picochat.tokenizer import EOS_TOKEN, IM_END

VOCAB = 40


class FakeTok:
    """Minimal tokenizer stand-in: ids 1/2 are the ChatML/EOS stop tokens."""

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def encode_single_token(self, token):
        return {IM_END: 1, EOS_TOKEN: 2}[token]


def _model(seed=0):
    torch.manual_seed(seed)
    return TransformerLM(
        vocab_size=VOCAB, d_model=32, n_heads=4, n_layers=2, max_seq_len=64
    )


def test_group_advantages_zero_mean_and_degenerate():
    adv = grpo.group_advantages(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert abs(float(adv.mean())) < 1e-5
    assert float(adv.std()) > 0
    # a group where every rollout scored the same carries no signal
    flat = grpo.group_advantages(torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(flat, torch.zeros(3))


def test_token_logprobs_shape_and_values():
    m = _model().eval()
    seqs = torch.randint(0, VOCAB, (3, 7))
    lp = grpo.token_logprobs(m, seqs)
    assert lp.shape == (3, 6)
    assert (lp <= 0).all()  # log-probs are non-positive


def test_grpo_loss_is_finite_and_kl_nonneg():
    torch.manual_seed(0)
    b, t = 4, 5
    policy = torch.randn(b, t) * 0.1
    ref = torch.randn(b, t) * 0.1
    old = policy.detach()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.ones(b, t)
    loss, metrics = grpo.grpo_loss(policy, ref, old, adv, mask, kl_coef=0.1)
    assert torch.isfinite(loss)
    assert metrics["kl"] >= 0.0
    # ratio is exactly 1 on-policy (old == policy)
    assert abs(metrics["ratio"] - 1.0) < 1e-5


def test_rollout_respects_stop_and_budget():
    m = _model().eval()
    cfg = SamplingConfig(temperature=1.0, max_new_tokens=6)
    resp = grpo.rollout(m, [3, 4, 5], group_size=5, cfg=cfg, stop_ids={1, 2})
    assert len(resp) == 5
    for r in resp:
        assert len(r) <= 6  # budget
        assert 1 not in r and 2 not in r  # stop tokens are excluded


def _module(max_steps=3):
    policy = _model(seed=1)
    reference = _model(seed=2)
    reward_model = RewardModel(judge=MockJudge(target_len=10))
    return grpo.GRPOModule(
        policy,
        reference,
        reward_model,
        pad_idx=0,
        tokenizer=FakeTok(),
        group_size=4,
        temperature=1.0,
        max_new_tokens=8,
        optimizer="adamw",
        lr=1e-3,
        warmup_steps=0,
        max_steps=max_steps,
        model_config={"max_seq_len": 64},
    )


def test_grpo_training_updates_policy_not_reference():
    module = _module(max_steps=3)
    before = copy.deepcopy(module.model.state_dict())
    ref_before = copy.deepcopy(module.reference.state_dict())

    data = [
        {"prompt_ids": [3, 4, 5], "task": None, "prompt_str": "solve it"}
        for _ in range(4)
    ]
    loader = DataLoader(data, batch_size=2, collate_fn=grpo.grpo_collate)

    trainer = L.Trainer(
        max_steps=3,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, loader)

    # policy moved, reference frozen, nothing exploded
    moved = any(
        not torch.allclose(before[k], v) for k, v in module.model.state_dict().items()
    )
    assert moved, "policy parameters did not update"
    for k, v in module.reference.state_dict().items():
        assert torch.allclose(ref_before[k], v), f"reference changed at {k}"
    for v in module.model.state_dict().values():
        assert torch.isfinite(v).all()


def test_reference_excluded_from_state_dict_and_optimizer():
    module = _module()
    # frozen reference is not a registered submodule -> not in the checkpoint
    assert not any(k.startswith("reference") for k in module.state_dict())
    # ...and not handed to the optimizer
    (opt,) = module.configure_optimizers()
    opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) not in opt_params for p in module.reference.parameters())
