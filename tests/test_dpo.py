"""DPO: the loss math, the pair dataset/collate, and a DPOModule step on a
tiny model -- all offline."""

import json

import pytest
import torch

from picochat.model import TransformerLM
from picochat.tokenizer import (
    PAD_TOKEN,
    SPECIAL_TOKENS,
    load_tokenizer,
    train_tokenizer,
)
from picochat.training.dpo import (
    DPOModule,
    PreferenceDataset,
    dpo_collate,
    dpo_loss,
    sequence_logprobs,
)


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    corpus = ["hello world this is a preference pair example reply"] * 40
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    train_tokenizer(
        iter(corpus), vocab_size=350, save_as=path, special_tokens=SPECIAL_TOKENS
    )
    return load_tokenizer(path)


def _pairs_file(tmp_path, n=4):
    path = tmp_path / "pairs.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": f"question {i}"}],
                        "chosen": "this is a helpful reply",
                        "rejected": "bad",
                    }
                )
                + "\n"
            )
    return path


# ---------------------------------------------------------------------------
# loss math
# ---------------------------------------------------------------------------
def test_dpo_loss_at_init_is_log2():
    # policy == reference -> margin 0 -> loss = -log sigmoid(0) = log 2
    z = torch.zeros(3)
    loss, metrics = dpo_loss(z, z, z, z)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert metrics["margin"] == 0.0


def test_dpo_loss_falls_as_margin_grows_and_beta_scales():
    z = torch.zeros(2)
    small, _ = dpo_loss(torch.tensor([1.0, 1.0]), z, z, z, beta=0.1)
    large, m = dpo_loss(torch.tensor([5.0, 5.0]), z, z, z, beta=0.1)
    assert large.item() < small.item() < torch.log(torch.tensor(2.0)).item()
    assert m["accuracy"] == 1.0
    sharp, _ = dpo_loss(torch.tensor([1.0, 1.0]), z, z, z, beta=1.0)
    assert sharp.item() < small.item()  # higher beta = sharper pressure


def test_dpo_loss_penalizes_preferring_rejected():
    z = torch.zeros(2)
    # policy raised the REJECTED side's logprob relative to the reference
    loss, metrics = dpo_loss(z, torch.tensor([2.0, 2.0]), z, z)
    assert loss.item() > torch.log(torch.tensor(2.0)).item()
    assert metrics["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# sequence log-probs
# ---------------------------------------------------------------------------
def test_sequence_logprobs_masks_prompt_and_padding():
    torch.manual_seed(0)
    lm = TransformerLM(vocab_size=50, d_model=32, n_heads=4, n_layers=2).eval()
    ids = torch.randint(1, 50, (1, 10))
    labels = torch.full_like(ids, -100)
    labels[:, 4:8] = ids[:, 4:8]  # only positions 4..7 are targets
    lp = sequence_logprobs(lm, ids, labels)
    assert lp.shape == (1,)
    # changing an UNMASKED-target-free region's label mask widens the sum
    labels2 = labels.clone()
    labels2[:, 8] = ids[:, 8]
    lp2 = sequence_logprobs(lm, ids, labels2)
    assert lp2.item() < lp.item()  # one more (negative) logprob term


# ---------------------------------------------------------------------------
# dataset / collate / module
# ---------------------------------------------------------------------------
def test_preference_dataset_and_collate(tok, tmp_path):
    pad = tok.encode_single_token(PAD_TOKEN)
    ds = PreferenceDataset(_pairs_file(tmp_path), tok, max_length=256, pad_id=pad)
    assert len(ds) == 4
    batch = dpo_collate([ds[0], ds[1]], pad_id=pad)
    assert batch["chosen_ids"].shape == batch["chosen_labels"].shape
    # masked/padding positions are -100, real targets echo the ids
    labels = batch["chosen_labels"]
    ids = batch["chosen_ids"]
    real = labels != -100
    assert real.any()
    assert torch.equal(labels[real], ids[real])
    assert (labels[~real] == -100).all()


def test_dpo_module_step_moves_policy_only(tok, tmp_path):
    torch.manual_seed(0)
    pad = tok.encode_single_token(PAD_TOKEN)
    lm = TransformerLM(vocab_size=tok.n_vocab, d_model=32, n_heads=4, n_layers=2)
    ref = TransformerLM(vocab_size=tok.n_vocab, d_model=32, n_heads=4, n_layers=2)
    ref.load_state_dict(lm.state_dict())
    module = DPOModule(lm, ref, pad_idx=pad, beta=0.1, tokenizer=tok)

    ds = PreferenceDataset(_pairs_file(tmp_path), tok, max_length=256, pad_id=pad)
    batch = dpo_collate([ds[0], ds[1]], pad_id=pad)
    loss, metrics = module._loss(batch)
    # identical policy/reference at init: exactly the log 2 fixed point
    assert loss.item() == pytest.approx(0.6931, abs=1e-3)
    loss.backward()
    assert any(p.grad is not None for p in module.model.parameters())
    # the reference is frozen and outside the optimizer/state_dict
    assert all(not p.requires_grad for p in module.reference.parameters())
    assert not any(k.startswith("reference") for k in module.state_dict())
    muon, adam = module._muon_param_split()
    opt_ids = {id(p) for p in muon} | {id(p) for g in adam for p in g["params"]}
    assert not any(id(p) in opt_ids for p in module.reference.parameters())
