"""LightningModule scaffolding shared by GPT (pretraining, MTP) and SFTModule
(supervised fine-tuning): Muon/AdamW optimizer wiring, the warmup+cosine LR
schedule applied by hand under manual optimization, and greedy KV-cache
generation. Both subclasses differ only in how they compute the loss (MTP's
multi-head, memory-optimized backward vs. a single next-token cross-entropy),
so that part stays in each class.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from picochat.optim import Muon


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
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        # "muon" (default) or "adamw". With muon, `lr`/`betas` still apply --
        # to the embedded AdamW that handles the params Muon skips.
        self.optimizer_name = optimizer
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.grad_clip = grad_clip
        self.accumulate = accumulate
        # Set by configure_optimizers: the per-group base LR the schedule scales.
        self._base_lrs: list[float] = []
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

    def _muon_param_groups(self) -> list[dict]:
        # Muon orthogonalizes matrix-shaped *hidden* weights. The embedding and
        # lm heads (input/output layers, per the Muon authors) and 1-dim params
        # (biases) go to the embedded AdamW instead, keeping the same decay
        # split as _param_groups: no decay for embeddings/1-dim, decay for the
        # lm-head matrices. Everything else -- attention/FFN projections, the
        # router, and the fused 3D MoE expert weights (flattened inside Muon.step)
        # -- is optimized by Muon.
        embed_ids = {
            id(p)
            for m in self.model.modules()
            if isinstance(m, nn.Embedding)
            for p in m.parameters()
        }
        head_ids = {id(p) for head in self.model.lmheads for p in head.parameters()}
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
        return [
            dict(
                params=muon,
                use_muon=True,
                lr=self.muon_lr,
                momentum=self.muon_momentum,
                weight_decay=self.weight_decay,
            ),
            dict(
                params=adam_decay,
                use_muon=False,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            ),
            dict(
                params=adam_no_decay,
                use_muon=False,
                lr=self.lr,
                betas=self.betas,
                weight_decay=0.0,
            ),
        ]

    def configure_optimizers(self):
        if self.optimizer_name == "muon":
            optimizer = Muon(self._muon_param_groups())
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self._param_groups(), lr=self.lr, betas=self.betas
            )
        else:
            raise ValueError(
                f"unknown optimizer '{self.optimizer_name}'. choices: muon, adamw"
            )
        # Under manual optimization the LR schedule is applied by hand in
        # _apply_lr (Lightning does not step schedulers for us here), so we just
        # remember each group's base LR and return the bare optimizer.
        self._base_lrs = [group["lr"] for group in optimizer.param_groups]
        return optimizer

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

    def _apply_lr(self, opt) -> None:
        # Scale each param group's base LR by the warmup/cosine schedule. Keyed on
        # global_step (optimizer steps), matching the old "interval: step"
        # scheduler. When max_steps is unknown, keep LR constant.
        if self.max_steps is None:
            return
        scale = self._lr_lambda(self.trainer.global_step)
        for base, group in zip(self._base_lrs, opt.param_groups):
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
        opt = self.optimizers()
        self._apply_lr(opt)
        if self.grad_clip:
            self.clip_gradients(
                opt, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm"
            )
        opt.step()
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
