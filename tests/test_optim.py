import pytest
import torch

from picochat.model.gpt import GPT, TransformerLM
from picochat.optim import Muon, zeropower_via_newtonschulz5


def test_newtonschulz_orthogonalizes():
    # Newton-Schulz drives every singular value toward ~1 (the coefficients
    # trade exactness for speed, so allow a generous band around it).
    torch.manual_seed(0)
    for shape in ((16, 32), (32, 8)):  # wide and tall
        X = zeropower_via_newtonschulz5(torch.randn(*shape)).float()
        assert X.shape == shape
        s = torch.linalg.svdvals(X)
        assert ((s > 0.4) & (s < 1.4)).all()


def test_muon_param_split_covers_everything_once():
    lm = TransformerLM(
        vocab_size=40,
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_experts=4,
        d_expert=16,
        n_lmheads=2,
    )
    gpt = GPT(lm, pad_idx=0)
    muon_group, adam_decay, adam_no_decay = gpt._muon_param_groups()
    muon_ids = {id(p) for p in muon_group["params"]}
    decay_ids = {id(p) for p in adam_decay["params"]}
    no_decay_ids = {id(p) for p in adam_no_decay["params"]}

    # fused MoE weights (router 2D + experts 3D) are Muon-optimized
    moe = lm.transformer.layers[0].moe
    for w in (moe.weight_router, moe.weight_up, moe.weight_gate, moe.weight_down):
        assert id(w) in muon_ids
    # only matrix-shaped params reach Muon
    assert all(p.ndim >= 2 for p in muon_group["params"])
    # embedding (no decay) and every lm head (decay) go to the embedded AdamW
    assert id(lm.embed.weight) in no_decay_ids
    for head in lm.lmheads:
        assert id(head.weight) in decay_ids
    # each trainable param lands in exactly one group
    all_ids = {id(p) for p in lm.parameters() if p.requires_grad}
    assert muon_ids | decay_ids | no_decay_ids == all_ids
    assert len(muon_ids) + len(decay_ids) + len(no_decay_ids) == len(all_ids)


def test_muon_step_updates_moe_and_preserves_shapes():
    torch.manual_seed(0)
    lm = TransformerLM(
        vocab_size=40, d_model=32, n_heads=4, n_layers=2, n_experts=4, d_expert=16
    )
    gpt = GPT(lm, pad_idx=0, compile=False)
    opt = gpt.configure_optimizers()
    moe = lm.transformer.layers[0].moe
    attn = lm.transformer.layers[0].attn
    shapes = {name: p.shape for name, p in lm.named_parameters()}
    before_up = moe.weight_up.detach().clone()
    before_q = attn.proj_q.weight.detach().clone()

    gpt.train()
    gpt._loss(torch.randint(1, 40, (2, 8))).backward()
    opt.step()

    for name, p in lm.named_parameters():
        assert p.shape == shapes[name]  # flatten/reshape round-trips exactly
        assert torch.isfinite(p).all()
    # the 3D expert weight and a plain 2D hidden matrix both moved
    assert not torch.allclose(moe.weight_up, before_up)
    assert not torch.allclose(attn.proj_q.weight, before_q)


def test_muon_overfits_single_batch():
    torch.manual_seed(0)
    lm = TransformerLM(vocab_size=40, d_model=32, n_heads=4, n_layers=2)
    gpt = GPT(lm, pad_idx=0, compile=False)
    opt = gpt.configure_optimizers()
    batch = torch.randint(1, 40, (2, 8))
    gpt.train()
    first = gpt._loss(batch).item()
    for _ in range(30):
        opt.zero_grad()
        loss = gpt._loss(batch)
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_muon_group_requires_flag():
    with pytest.raises(ValueError):
        Muon([dict(params=[torch.nn.Parameter(torch.randn(4, 4))])])
