"""SFT (supervised fine-tuning) LightningModule.

Kept separate from GPT (pretraining) rather than folded into it: GPT trains on
a single packed token stream (input == target, shifted internally), while SFT
trains against pre-computed (input_ids, labels) pairs from picochat.data.sft,
where labels already carry the loss mask (see
picochat.data.sft.encode_conversation). The two build their loss from different
batch shapes; optimizer/LR-schedule/generation code is shared via
LMTrainerMixin instead.
"""

import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from picochat.model.lm_module import LMTrainerMixin, can_compile
from picochat.model.transformer import TransformerLM


class SFTModule(LMTrainerMixin, L.LightningModule):
    def __init__(
        self,
        transformer_lm: TransformerLM,
        pad_idx: int,
        lr: float = 1e-5,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        optimizer: str = "muon",
        muon_lr: float = 0.005,
        muon_momentum: float = 0.95,
        warmup_steps: int = 100,
        max_steps: int | None = None,
        min_lr_ratio: float = 0.1,
        grad_clip: float | None = 1.0,
        accumulate: int = 1,
        compile: bool | None = None,
        tokenizer=None,
        model_config: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters("model_config")
        self.model = transformer_lm
        self.pad_idx = pad_idx
        self._init_trainer(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            optimizer=optimizer,
            muon_lr=muon_lr,
            muon_momentum=muon_momentum,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
            grad_clip=grad_clip,
            accumulate=accumulate,
            tokenizer=tokenizer,
        )
        # Manual optimization: mirrors GPT so the two share the same LR
        # schedule / gradient-accumulation step (see LMTrainerMixin._optimizer_step).
        self.automatic_optimization = False
        self.compile = can_compile() if compile is None else compile
        self._forward = torch.compile(self.model) if self.compile else self.model

    def _loss(self, input_ids: Tensor, labels: Tensor) -> Tensor:
        # Next-token prediction: position i's logits predict token i+1, so
        # compare against labels shifted left by one -- labels is already
        # input-aligned (see picochat.data.sft), not pre-shifted.
        logits = self._forward(input_ids)[:, :-1]
        targets = labels[:, 1:]
        return F.cross_entropy(
            rearrange(logits, "b l v -> (b l) v"),
            rearrange(targets, "b l -> (b l)"),
            ignore_index=self.pad_idx,
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._loss(batch["input_ids"], batch["labels"])
        (loss / self.accumulate).backward()
        self._optimizer_step(batch_idx)
        self.log("train_loss", loss)
        self.log("loss", loss, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._loss(batch["input_ids"], batch["labels"])
        self.log("val_loss", loss, prog_bar=True)
        return loss
