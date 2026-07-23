"""Multi-GPU (DDP) behaviors, verified on CPU with the gloo backend: the MoE
load-balancing bias must follow the *global* batch's load and stay identical
across ranks, gradient accumulation must suppress DDP's per-microbatch
all-reduce, and the sampler/seed/Trainer wiring the training scripts use must
survive a real 2-process Lightning fit. Workers for mp.spawn live at module
level (they must be picklable)."""

from contextlib import nullcontext
from types import SimpleNamespace

import lightning as L
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset

from picochat.data.dataloader import PretrainDataModule
from picochat.model import MixtureOfExperts, TransformerLM
from picochat.training import GPT, LMTrainerMixin


def _init_pg(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )


# ---------------------------------------------------------------------------
# MoE load-balancing bias under DDP
# ---------------------------------------------------------------------------
def _bias_worker(rank: int, world_size: int, init_file: str) -> None:
    _init_pg(rank, world_size, init_file)
    try:
        torch.manual_seed(0)  # identical experts on both ranks, like DDP
        moe = MixtureOfExperts(8, d_hidden=16, n_experts=4, n_active=2)
        # Rank-local loads that sum to the global [3, 1, 1, 3] (mean 2), so the
        # per-rank signs would differ from the global ones -- the assertion
        # below only holds if the counts really were all-reduced.
        local = {0: [3.0, 0.0, 1.0, 0.0], 1: [0.0, 1.0, 0.0, 3.0]}[rank]
        moe._pending_load.copy_(torch.tensor(local))
        moe.apply_bias_update()
        expected = moe.bias_update_rate * torch.tensor([-1.0, 1.0, 1.0, -1.0])
        assert torch.allclose(moe.expert_bias, expected), (rank, moe.expert_bias)
        # and therefore identical on every rank (no rank-0 broadcast needed)
        gathered = [torch.zeros_like(moe.expert_bias) for _ in range(world_size)]
        dist.all_gather(gathered, moe.expert_bias)
        assert torch.equal(gathered[0], gathered[1])
    finally:
        dist.destroy_process_group()


def test_moe_bias_update_follows_global_load_across_ranks(tmp_path):
    mp.spawn(_bias_worker, args=(2, str(tmp_path / "init")), nprocs=2, join=True)


# ---------------------------------------------------------------------------
# Gradient accumulation vs DDP gradient sync
# ---------------------------------------------------------------------------
def test_grad_sync_context_suppresses_ddp_sync_while_accumulating(tmp_path):
    # world_size=1 gloo still builds a real DDP wrapper, whose
    # require_backward_grad_sync flag is what no_sync() toggles.
    _init_pg(0, 1, str(tmp_path / "init"))
    try:
        ddp = DistributedDataParallel(torch.nn.Linear(4, 4))
        m = LMTrainerMixin()
        m.accumulate = 2
        m._trainer = SimpleNamespace(strategy=SimpleNamespace(model=ddp))
        # cycle boundary (batch_idx 1 of 2): the backward must sync
        assert isinstance(m._grad_sync_context(1), nullcontext)
        # mid-accumulation: suppressed
        ctx = m._grad_sync_context(0)
        assert not isinstance(ctx, nullcontext)
        with ctx:
            assert ddp.require_backward_grad_sync is False
        assert ddp.require_backward_grad_sync is True  # restored on exit
    finally:
        dist.destroy_process_group()


def test_grad_sync_context_noop_outside_ddp():
    # No Trainer attached (unit tests, plain modules): always a nullcontext.
    m = LMTrainerMixin()
    m.accumulate = 4
    assert isinstance(m._grad_sync_context(0), nullcontext)


# ---------------------------------------------------------------------------
# Validation generation samples run on rank 0 only
# ---------------------------------------------------------------------------
def test_generation_sample_skipped_on_nonzero_rank():
    # On ranks != 0 the logger's experiment is a no-op DummyExperiment, so
    # without the guard every rank would pay for the slow greedy decode just
    # to discard the text. _generate raising proves the early return.
    lm = TransformerLM(vocab_size=16, d_model=8, n_heads=2, n_layers=1)
    gpt = GPT(lm, pad_idx=0, compile=False)
    gpt._generate = lambda *a, **k: pytest.fail("decode ran on a nonzero rank")
    gpt._trainer = SimpleNamespace(is_global_zero=False)
    gpt._log_generation_sample(torch.zeros(1, 8, dtype=torch.long), batch_idx=0)


# ---------------------------------------------------------------------------
# End-to-end: 2-process CPU DDP fit with the scripts' Trainer wiring
# ---------------------------------------------------------------------------
class _PackedRows(Dataset):
    """Synthetic packed pretraining rows: BOS followed by random tokens."""

    def __init__(self, vocab: int, seq_len: int, n: int, bos: int):
        g = torch.Generator().manual_seed(0)
        self.rows = torch.randint(3, vocab, (n, seq_len), generator=g)
        self.rows[:, 0] = bos

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def test_two_process_ddp_fit_with_rank_aware_sampler(tmp_path):
    # The full integration the training scripts rely on: ddp (2 CPU procs) +
    # use_distributed_sampler=False + the seeded chunked sampler + gradient
    # accumulation + an MoE model (exercising the bias all-reduce inside a
    # real training forward). Passing means Lightning left our sampler in
    # place (its wrapper would choke the run) and every rank stepped in sync.
    torch.manual_seed(0)
    lm = TransformerLM(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_layers=2,
        max_seq_len=32,
        n_experts=4,
        d_expert=8,
        layers_per_block=1,
        grad_checkpoint=False,
    )
    gpt = GPT(
        lm,
        pad_idx=0,
        bos_idx=1,
        optimizer="adamw",
        warmup_steps=1,
        max_steps=2,
        accumulate=2,
        compile=False,
    )
    dm = PretrainDataModule(
        _PackedRows(64, 17, n=64, bos=1),
        # A val set too: exercises the datamodule's own DistributedSampler and
        # the sync_dist'd val_loss reduction across ranks in a real fit.
        _PackedRows(64, 17, n=8, bos=1),
        batch_size=2,
        num_workers=0,
        seed=7,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp_spawn",
        max_steps=2,
        limit_val_batches=2,
        use_distributed_sampler=False,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # ddp_spawn trains in child processes, so the parent's trainer.global_step
    # stays 0; what does come back is the trained weights (Lightning restores
    # rank 0's state into the main-process model). Assert on those instead.
    before = {k: v.clone() for k, v in gpt.model.state_dict().items()}
    trainer.fit(gpt, datamodule=dm)
    assert trainer.state.finished
    after = gpt.model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
