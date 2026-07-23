"""Training-time benchmark evaluation: a Lightning callback that runs the
likelihood-based MCQ benchmarks (picochat.evals.tasks) on the in-training
model every N optimizer steps and logs acc / acc_norm next to the loss
curves, so data-mixture or LR problems show up mid-run instead of after a
full training + manual base_eval cycle.

Kept deliberately lightweight: a capped `limit` per task (a few hundred items
tracks the trend; the full set is for final numbers), rank 0 only under DDP
(the eval calls the bare module, so no collectives are involved -- and
eval-mode MoE skips the load-balancing all-reduce staging), and the model is
flipped back to train mode afterwards. Task data comes from the HF Hub, so
the first evaluation of a run may download datasets.
"""

from __future__ import annotations

import lightning as L
import torch

from picochat.evals.tasks import evaluate_task
from picochat.tokenizer import Tokenizer


class BenchmarkEvalCallback(L.Callback):
    """Run MCQ benchmarks during training and log `bench/<task>/acc(_norm)`.

    tasks: names from picochat.evals.tasks.TASKS.
    limit: items per task per evaluation (None = full set; keep it capped).
    every_n_steps: evaluate when global_step crosses each multiple.
    chat: render items as ChatML user turns (SFT stages) vs plain text (base).
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        tasks: list[str] | tuple[str, ...] = ("hellaswag",),
        limit: int | None = 200,
        every_n_steps: int = 1000,
        chat: bool = False,
        batch_size: int = 16,
        max_len: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.tasks = list(tasks)
        self.limit = limit
        self.every_n_steps = every_n_steps
        self.chat = chat
        self.batch_size = batch_size
        self.max_len = max_len
        self._last_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0 or step == self._last_step:
            return
        self._last_step = step  # global_step repeats across accumulate microbatches
        if not trainer.is_global_zero:
            return
        self._evaluate(trainer, pl_module)

    @torch.no_grad()
    def _evaluate(self, trainer, pl_module) -> None:
        model = pl_module.model  # the bare TransformerLM: no DDP sync involved
        was_training = model.training
        model.eval()
        try:
            for task in self.tasks:
                try:
                    result = evaluate_task(
                        model,
                        self.tokenizer,
                        task,
                        chat=self.chat,
                        limit=self.limit,
                        batch_size=self.batch_size,
                        max_len=self.max_len,
                        device=pl_module.device,
                    )
                except Exception as e:  # a Hub hiccup must not kill the run
                    print(f"benchmark eval '{task}' failed: {e}", flush=True)
                    continue
                pl_module.log(f"bench/{task}/acc", result["acc"], rank_zero_only=True)
                pl_module.log(
                    f"bench/{task}/acc_norm", result["acc_norm"], rank_zero_only=True
                )
                print(
                    f"[bench @ step {trainer.global_step}] {task}: "
                    f"acc {result['acc']:.3f}  acc_norm {result['acc_norm']:.3f}",
                    flush=True,
                )
        finally:
            model.train(was_training)


def benchmark_callback_from_config(
    trainer_cfg: dict, tokenizer: Tokenizer, chat: bool
) -> BenchmarkEvalCallback | None:
    """Build the callback from a stage config's `trainer.benchmark_eval`
    section (None when absent):

        trainer:
            benchmark_eval:
                tasks: [hellaswag, arc_easy]
                limit: 200
                every_n_steps: 1000
    """
    cfg = trainer_cfg.get("benchmark_eval")
    if not cfg:
        return None
    return BenchmarkEvalCallback(
        tokenizer=tokenizer,
        tasks=cfg.get("tasks", ["hellaswag"]),
        limit=cfg.get("limit", 200),
        every_n_steps=cfg.get("every_n_steps", 1000),
        chat=cfg.get("chat", chat),
        batch_size=cfg.get("batch_size", 16),
        max_len=cfg.get("max_len", 4096),
    )


class NaNGuardCallback(L.Callback):
    """Stop training the moment the loss goes non-finite (NaN/Inf), instead of
    burning hours of GPU on a diverged run. A single spike is usually
    unrecoverable at this scale (the optimizer step already poisoned the
    weights), so the default patience of 1 halts immediately; raise it to
    tolerate `patience` consecutive bad steps before stopping.

    Sets trainer.should_stop, which ends training gracefully after the current
    batch -- Lightning still writes the last checkpoint, so the pre-divergence
    state (from the previous good checkpoint) stays available.
    """

    def __init__(self, patience: int = 1):
        self.patience = patience
        self._bad = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # manual-optimization training_step returns the loss tensor directly;
        # the automatic loop wraps it as {"loss": ...}.
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            return
        finite = torch.isfinite(torch.as_tensor(loss)).all().item()
        if finite:
            self._bad = 0
            return
        self._bad += 1
        print(
            f"NaNGuard: non-finite loss at global_step {trainer.global_step} "
            f"({self._bad}/{self.patience})",
            flush=True,
        )
        if self._bad >= self.patience:
            print("NaNGuard: stopping training (diverged).", flush=True)
            trainer.should_stop = True
