"""Optimizer wiring tests: torch.optim.Muon for hidden matrices + AdamW for
the rest, split by LMTrainerMixin._muon_param_split (see picochat/trainer.py)."""

import torch

from picochat.model import TransformerLM
from picochat.training import GPT


def _moe_gpt() -> GPT:
    lm = TransformerLM(
        vocab_size=40,
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_experts=4,
        d_expert=16,
    )
    return GPT(lm, pad_idx=0, compile=False)


def test_muon_param_split_covers_everything_once():
    gpt = _moe_gpt()
    lm = gpt.model
    muon_params, (adam_decay, adam_no_decay) = gpt._muon_param_split()
    muon_ids = {id(p) for p in muon_params}
    decay_ids = {id(p) for p in adam_decay["params"]}
    no_decay_ids = {id(p) for p in adam_no_decay["params"]}

    # fused MoE weights (router + flattened 2D experts) are Muon-optimized
    moe = lm.transformer.layers[0].moe
    for w in (
        moe.weight_router,
        moe.bank.weight_up,
        moe.bank.weight_gate,
        moe.bank.weight_down,
    ):
        assert id(w) in muon_ids
    # torch.optim.Muon accepts exactly 2D params, nothing else
    assert all(p.ndim == 2 for p in muon_params)
    # embedding (no decay) and the lm head (decay) go to AdamW
    assert id(lm.embed.weight) in no_decay_ids
    assert id(lm.lmhead.weight) in decay_ids
    assert adam_decay["weight_decay"] > 0
    assert adam_no_decay["weight_decay"] == 0.0
    # each trainable param lands in exactly one group
    all_ids = {id(p) for p in lm.parameters() if p.requires_grad}
    assert muon_ids | decay_ids | no_decay_ids == all_ids
    assert len(muon_ids) + len(decay_ids) + len(no_decay_ids) == len(all_ids)


def test_configure_optimizers_builds_torch_muon_and_adamw():
    gpt = _moe_gpt()
    muon_opt, adam_opt = gpt.configure_optimizers()
    assert isinstance(muon_opt, torch.optim.Muon)
    assert isinstance(adam_opt, torch.optim.AdamW)
    # muon_weight_decay is independent of weight_decay (see
    # LMTrainerMixin._init_trainer): reusing weight_decay for Muon would
    # over-decay it since muon_lr sits an order of magnitude+ above lr, so
    # torch.optim.Muon's own decoupled decay (param *= 1 - lr * weight_decay)
    # would otherwise be that much stronger than AdamW's.
    assert muon_opt.param_groups[0]["weight_decay"] == gpt.muon_weight_decay
    assert gpt.muon_weight_decay != gpt.weight_decay
    # the pair covers the whole model
    n_opt = sum(
        p.numel()
        for opt in (muon_opt, adam_opt)
        for g in opt.param_groups
        for p in g["params"]
    )
    assert n_opt == sum(p.numel() for p in gpt.model.parameters())
    # base LRs captured per optimizer for the manual LR schedule
    assert gpt._base_lrs == [
        [g["lr"] for g in muon_opt.param_groups],
        [g["lr"] for g in adam_opt.param_groups],
    ]


def test_muon_step_updates_moe_and_preserves_shapes():
    torch.manual_seed(0)
    gpt = _moe_gpt()
    lm = gpt.model
    optimizers = gpt.configure_optimizers()
    moe = lm.transformer.layers[0].moe
    attn = lm.transformer.layers[0].attn
    shapes = {name: p.shape for name, p in lm.named_parameters()}
    before_up = moe.bank.weight_up.detach().clone()
    before_q = attn.proj_q.weight.detach().clone()
    before_embed = lm.embed.weight.detach().clone()

    gpt.train()
    gpt._loss(torch.randint(1, 40, (2, 8))).backward()
    for opt in optimizers:
        opt.step()

    for name, p in lm.named_parameters():
        assert p.shape == shapes[name]
        assert torch.isfinite(p).all()
    # a Muon param (2D expert weight), a plain hidden matrix and an AdamW
    # param (embedding) all moved
    assert not torch.allclose(moe.bank.weight_up, before_up)
    assert not torch.allclose(attn.proj_q.weight, before_q)
    assert not torch.allclose(lm.embed.weight, before_embed)


def test_muon_overfits_single_batch():
    torch.manual_seed(0)
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    gpt = GPT(lm, pad_idx=0, compile=False)
    optimizers = gpt.configure_optimizers()
    batch = torch.randint(1, 40, (2, 8))
    gpt.train()
    first = gpt._loss(batch).item()
    for _ in range(30):
        for opt in optimizers:
            opt.zero_grad()
        loss = gpt._loss(batch)
        loss.backward()
        for opt in optimizers:
            opt.step()
    assert loss.item() < first


def test_muon_param_split_with_nsa_layer():
    # Regression: a model deep enough to contain an NSA layer (block tail at
    # layers_per_block=4) must split cleanly. NSA's positional signal is
    # PartialRoPE (buffers, not params), so it adds nothing to the AdamW side;
    # earlier code referenced a nonexistent NSA.cmp_pos and crashed here.
    from picochat.model.sparse_attn import NativeSparseAttention

    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=4)
    nsa_layers = [m for m in lm.modules() if isinstance(m, NativeSparseAttention)]
    assert nsa_layers, "expected an NSA layer at n_layers=4"
    gpt = GPT(lm, pad_idx=0, compile=False)
    muon_params, (adam_decay, adam_no_decay) = gpt._muon_param_split()
    # every trainable param lands in exactly one group (NSA params included)
    all_ids = {id(p) for p in lm.parameters() if p.requires_grad}
    split_ids = (
        {id(p) for p in muon_params}
        | {id(p) for p in adam_decay["params"]}
        | {id(p) for p in adam_no_decay["params"]}
    )
    assert split_ids == all_ids
    # configure_optimizers actually builds the optimizers without raising
    opts = gpt.configure_optimizers()
    assert len(opts) == 2  # Muon + AdamW
