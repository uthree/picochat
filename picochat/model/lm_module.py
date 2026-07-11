"""LightningModule scaffolding shared by GPT (pretraining) and SFTModule
(supervised fine-tuning): Muon/AdamW optimizer wiring, the warmup+cosine LR
schedule applied by hand under manual optimization, and greedy KV-cache
generation. Both subclasses differ only in how they build the batch into a
next-token cross-entropy loss, so that part stays in each class.

The "muon" mode runs two optimizers side by side: torch.optim.Muon for the
matrix-shaped hidden weights and torch.optim.AdamW for the rest (embeddings,
lm head, 1-dim params) -- torch's Muon is Muon-only, unlike the previous
in-repo implementation that embedded its own AdamW. Both subclasses already
use manual optimization, so _optimizer_step owns the pair (see the
global_step note there).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


def can_compile() -> bool:
    """Whether torch.compile is likely to help in this environment.

    The inductor backend targets CUDA; on CPU/MPS it often falls back or errors,
    so we only enable it on CUDA. torch.compile itself is lazy (compiles on the
    first forward), so this just gates whether we wrap the model at all.
    """
    return hasattr(torch, "compile") and torch.cuda.is_available()


class LMTrainerMixin:
    """Expects `self.model` (a TransformerLM) to be set before any of these
    methods run. Call `_init_trainer` from the subclass's __init__ to set the
    constructor-derived attributes these methods read.
    """

    def _init_trainer(
        self,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        optimizer: str,
        muon_lr: float,
        muon_momentum: float,
        warmup_steps: int,
        max_steps: int | None,
        min_lr_ratio: float,
        grad_clip: float | None,
        accumulate: int,
        tokenizer=None,
        muon_weight_decay: float = 0.01,
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        # "muon" (default) or "adamw". With muon, `lr`/`betas` still apply --
        # to the AdamW that runs alongside it for the params Muon skips.
        self.optimizer_name = optimizer
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        # Separate from `weight_decay`: torch.optim.Muon's decay is decoupled
        # (param *= 1 - lr * weight_decay) exactly like AdamW's, but muon_lr is
        # tuned an order of magnitude (or more) above `lr` -- reusing
        # `weight_decay` as-is would make Muon's *effective* decay that many
        # times stronger than AdamW's for no reason. Kept independently tunable.
        self.muon_weight_decay = muon_weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.grad_clip = grad_clip
        self.accumulate = accumulate
        # Set by configure_optimizers: per optimizer, the per-group base LRs
        # the schedule scales.
        self._base_lrs: list[list[float]] = []
        # Optional tiktoken Encoding used to turn generated token ids back into
        # readable text (e.g. for TensorBoard generation samples).
        self.tokenizer = tokenizer

    def _param_groups(self) -> list[dict]:
        # Apply weight decay only to weights with 2+ dims. Exclude biases (1-dim)
        # and embeddings (rms_norm has no learnable params, so nothing to exclude there).
        embed_ids = {
            id(p)
            for m in self.model.modules()
            if isinstance(m, nn.Embedding)
            for p in m.parameters()
        }
        decay, no_decay = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or id(p) in embed_ids:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _muon_param_split(self) -> tuple[list, list[dict]]:
        # Muon orthogonalizes matrix-shaped *hidden* weights. The embedding and
        # lm heads (input/output layers, per the Muon authors) and 1-dim params
        # (biases) go to the AdamW running alongside it instead, keeping the
        # same decay split as _param_groups: no decay for embeddings/1-dim,
        # decay for the lm-head matrices. Everything else -- attention/FFN
        # projections, the router, and the fused MoE expert weights (stored 2D
        # exactly because torch.optim.Muon accepts nothing else, see
        # MixtureOfExperts) -- is optimized by Muon.
        embed_ids = {
            id(p)
            for m in self.model.modules()
            if isinstance(m, nn.Embedding)
            for p in m.parameters()
        }
        head_ids = {id(p) for p in self.model.lmhead.parameters()}
        muon, adam_decay, adam_no_decay = [], [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or id(p) in embed_ids:
                adam_no_decay.append(p)
            elif id(p) in head_ids:
                adam_decay.append(p)
            else:
                muon.append(p)
        return muon, [
            dict(params=adam_decay, weight_decay=self.weight_decay),
            dict(params=adam_no_decay, weight_decay=0.0),
        ]

    def configure_optimizers(self):
        if self.optimizer_name == "muon":
            muon_params, adam_groups = self._muon_param_split()
            # adjust_lr_fn defaults to "original" (sqrt(max(1, rows/cols))
            # update scaling), the same correction the previous in-repo Muon
            # applied.
            optimizers = [
                torch.optim.Muon(
                    muon_params,
                    lr=self.muon_lr,
                    momentum=self.muon_momentum,
                    weight_decay=self.muon_weight_decay,
                ),
                torch.optim.AdamW(adam_groups, lr=self.lr, betas=self.betas),
            ]
        elif self.optimizer_name == "adamw":
            optimizers = [
                torch.optim.AdamW(self._param_groups(), lr=self.lr, betas=self.betas)
            ]
        else:
            raise ValueError(
                f"unknown optimizer '{self.optimizer_name}'. choices: muon, adamw"
            )
        # Under manual optimization the LR schedule is applied by hand in
        # _apply_lr (Lightning does not step schedulers for us here), so we just
        # remember each group's base LR and return the bare optimizers.
        self._base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
        return optimizers

    def _lr_lambda(self, step: int) -> float:
        # Linear warmup -> cosine decay (down to min_lr_ratio).
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        if self.max_steps is None or step >= self.max_steps:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * coeff

    def _apply_lr(self, opts: list) -> None:
        # Scale each param group's base LR by the warmup/cosine schedule. Keyed on
        # global_step (optimizer steps), matching the old "interval: step"
        # scheduler. When max_steps is unknown, keep LR constant.
        if self.max_steps is None:
            return
        scale = self._lr_lambda(self.trainer.global_step)
        for base_lrs, opt in zip(self._base_lrs, opts):
            for base, group in zip(base_lrs, opt.param_groups):
                group["lr"] = base * scale

    def _optimizer_step(self, batch_idx: int) -> None:
        # Manual optimization: step once every `accumulate` microbatches, applying
        # the LR schedule and gradient clipping ourselves. No-op when not attached
        # to a Trainer (e.g. training_step called directly in a unit test) --
        # the loss backward has already populated .grad there.
        if getattr(self, "_trainer", None) is None:
            return
        if (batch_idx + 1) % self.accumulate != 0:
            return  # keep accumulating grads into .grad
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]  # Lightning unwraps a single optimizer
        self._apply_lr(opts)
        if self.grad_clip:
            # One global-norm clip over every parameter, matching the previous
            # single-optimizer behavior; self.clip_gradients per optimizer
            # would clip each subset against the threshold separately. (The
            # bf16-mixed precision used here has no grad scaler, so clipping
            # raw grads directly is safe.)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        # Step the first optimizer through its Lightning wrapper and the rest
        # on the raw optimizer: Lightning counts every wrapped .step() into
        # trainer.global_step, so stepping both Muon and AdamW through
        # wrappers would advance global_step twice per cycle, doubling the LR
        # schedule's clock and halving the effective max_steps. Checkpointing
        # is unaffected -- Lightning saves optimizer state from
        # trainer.optimizers, not from who called .step().
        opts[0].step()
        for opt in opts[1:]:
            opt.optimizer.step()
        for opt in opts:
            opt.zero_grad()

    @torch.no_grad()
    def _generate(self, prompt: Tensor, max_new_tokens: int) -> Tensor:
        """Greedy-decode `max_new_tokens` tokens after `prompt` (B, L) via KV cache."""
        # `pos` tracks the absolute decode position as a plain local int -- not
        # model state -- and is threaded through each call, same as `cache`.
        logits, cache, pos = self.model.decode(prompt)
        next_token = logits[:, -1:].argmax(dim=-1)
        out = [next_token]
        for _ in range(max_new_tokens - 1):
            logits, cache, pos = self.model.decode(next_token, cache, pos)
            next_token = logits[:, -1:].argmax(dim=-1)
            out.append(next_token)
        return torch.cat(out, dim=1)  # (B, max_new_tokens)

    def _decode_text(self, ids: Tensor) -> str:
        try:
            return self.tokenizer.decode(ids.tolist())
        except Exception:
            return "<decode error>"
