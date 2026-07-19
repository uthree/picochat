"""Training-side LightningModules for the model in gpt.py: GPT (pretraining)
and SFTModule (supervised fine-tuning), plus the LMTrainerMixin scaffolding
they share -- Muon/AdamW optimizer wiring, the warmup+cosine LR schedule
applied by hand under manual optimization, and greedy KV-cache generation.
(GRPOModule in picochat.grpo builds on the same mixin.)
The two modules differ only in how they build the batch into a next-token
cross-entropy loss, so that part stays in each class:

- GPT trains on a single packed token stream (input == target, shifted
  internally), deriving per-token document ids from <|begin_of_text|>.
- SFTModule trains against pre-computed (input_ids, labels) pairs from
  picochat.dataloader, where labels already carry the loss mask (see
  picochat.tokenizer.encode_conversation).

The "muon" mode runs two optimizers side by side: torch.optim.Muon for the
matrix-shaped hidden weights and torch.optim.AdamW for the rest (embeddings,
lm head, 1-dim params) -- torch's Muon is Muon-only, unlike the previous
in-repo implementation that embedded its own AdamW. Both subclasses already
use manual optimization, so _optimizer_step owns the pair (see the
global_step note there).
"""

import math
from contextlib import nullcontext

import lightning as L
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from picochat.gpt import TransformerLM
from picochat.presets import build_lm
from picochat.kernels import (
    fused_linear_cross_entropy,
    fused_linear_cross_entropy_available,
)


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
        fused_loss: bool = False,
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
        # Liger fused lm-head + cross-entropy (see picochat.kernels): the
        # single biggest memory lever at 128k vocab, but opt-in -- the
        # chunked kernel trades some step time for that memory on smaller
        # GPUs, so it should be a deliberate choice when memory-bound, not a
        # silent default. True insists (a loud error beats silently training
        # slower/differently than configured); the availability probe may hit
        # the Hub once.
        if fused_loss and not fused_linear_cross_entropy_available():
            raise RuntimeError(
                "fused_loss=True but the fused kernel is unavailable -- it "
                "needs CUDA, the `kernels` package (pip install "
                "picochat[kernels]) and the liger-kernels Hub repo (cached "
                "after the first fetch)"
            )
        self.fused_loss = fused_loss

    def _setup_compiled_forward(self, compile: bool | None) -> None:
        """Build self._forward, the trainable forward, torch.compiled when the
        environment supports it (`compile=None` -> auto). With fused_loss OR
        multi-token prediction it stops at the hidden states (the lm/MTP heads
        run in _next_token_loss); otherwise it is the full model up to the
        primary logits. The compiled handle shares parameters with self.model,
        which stays uncompiled so decode() runs eager. Stored via
        object.__setattr__ so nn.Module doesn't register it as a submodule --
        registered, it would duplicate every weight in state_dict under
        `_forward.*` (or `_forward._orig_mod.*` when compiled), making a
        checkpoint loadable only under the exact same compile setting.

        Call after _init_trainer (sets self.fused_loss) and after self.model."""
        self.compile = can_compile() if compile is None else compile
        self._trunk_hidden = self.fused_loss or self.model.n_mtp > 0
        trunk = self.model.encode if self._trunk_hidden else self.model
        object.__setattr__(
            self, "_forward", torch.compile(trunk) if self.compile else trunk
        )

    def _backward_and_step(self, loss: Tensor, batch_idx: int) -> None:
        """The shared manual-optimization step: scale for gradient accumulation,
        backward, then the optimizer/LR-schedule step (see _optimizer_step), and
        log train_loss plus the progress-bar loss. Subclasses compute `loss`
        their own way and may log extra metrics on top."""
        with self._grad_sync_context(batch_idx):
            (loss / self.accumulate).backward()
        self._optimizer_step(batch_idx)
        self.log("train_loss", loss)
        self.log("loss", loss, prog_bar=True, logger=False)  # for progress bar

    def _grad_sync_context(self, batch_idx: int):
        """DDP all-reduces gradients on every backward, but while accumulating
        microbatches only the cycle's last backward needs the reduction --
        grads keep accumulating locally either way, and _optimizer_step skips
        the non-boundary batches. Suppress the redundant all-reduces with
        DDP's no_sync() so an `accumulate: k` cycle pays for one gradient
        sync, not k. A no-op outside DDP (single device, or no Trainer)."""
        if (batch_idx + 1) % self.accumulate == 0:
            return nullcontext()
        trainer = getattr(self, "_trainer", None)
        model = getattr(trainer.strategy, "model", None) if trainer else None
        if isinstance(model, DistributedDataParallel):
            return model.no_sync()
        return nullcontext()

    def _embedding_param_ids(self) -> set[int]:
        """ids of the embedding parameters -- excluded from weight decay and,
        under Muon, routed to AdamW like the input/output layers."""
        return {
            id(p)
            for m in self.model.modules()
            if isinstance(m, nn.Embedding)
            for p in m.parameters()
        }

    def _param_groups(self) -> list[dict]:
        # Apply weight decay only to weights with 2+ dims. Exclude biases (1-dim)
        # and embeddings (rms_norm has no learnable params, so nothing to exclude there).
        embed_ids = self._embedding_param_ids()
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
        embed_ids = self._embedding_param_ids()
        # The lm head is the model's output projection -> AdamW, not Muon (Muon
        # skips input/output layers). The MTP heads' transforms are hidden
        # d_model x d_model matrices (they reuse this same output projection), so
        # they go to Muon like every other hidden weight.
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

    def _head_ce(self, hidden: Tensor, weight: Tensor, target_ids: Tensor) -> Tensor:
        """Cross-entropy of a single output head (weight: (vocab, d_model)) over
        already-aligned hidden states and target ids. Uses the fused lm-head +
        cross-entropy kernel on CUDA (the (b*l, vocab) logits are never
        materialized -- the big memory lever at 128k vocab), a plain
        F.cross_entropy otherwise."""
        hidden = hidden.reshape(-1, hidden.shape[-1])
        target_ids = target_ids.reshape(-1)
        if self.fused_loss and hidden.is_cuda:
            return fused_linear_cross_entropy(
                hidden, weight, target_ids, ignore_index=self.pad_idx
            )
        return F.cross_entropy(
            hidden @ weight.T, target_ids, ignore_index=self.pad_idx
        )

    def _next_token_loss(self, input_ids: Tensor, targets: Tensor, masks) -> Tensor:
        """Next-token cross-entropy: position i's prediction is scored against
        targets[i+1] (`targets` is input-aligned, not pre-shifted; it is
        simply `input_ids` for pretraining, the masked labels for SFT).
        Positions whose target is pad_idx are ignored.

        With multi-token prediction (model.n_mtp > 0) the primary next-token
        loss is joined by one auxiliary cross-entropy per MTP head: head j reads
        the same hidden state and predicts the token at offset 2+j, weighted by
        `mtp_weight`. Otherwise this is the plain single-head next-token loss.

        When `self._trunk_hidden` (fused_loss or MTP) `self._forward` is the model
        trunk up to the final hidden states and the head matmuls happen here (see
        _head_ce); otherwise it is the full model and this is a plain loss over
        its primary logits.
        """
        if not self._trunk_hidden:
            logits = self._forward(input_ids, masks=masks)[:, :-1]
            return F.cross_entropy(
                rearrange(logits, "b l v -> (b l) v"),
                rearrange(targets[:, 1:], "b l -> (b l)"),
                ignore_index=self.pad_idx,
            )
        hidden = self._forward(input_ids, masks=masks)
        # Primary head: offset +1. hidden[:, :-1] predicts targets[:, 1:].
        loss = self._head_ce(hidden[:, :-1], self.model.lmhead.weight, targets[:, 1:])
        if self.model.n_mtp == 0:
            return loss
        return loss + self._mtp_loss(hidden, targets)

    def _mtp_loss(self, hidden: Tensor, targets: Tensor) -> Tensor:
        """Auxiliary multi-token-prediction loss (assumes model.n_mtp > 0): head
        j reads the shared hidden state and predicts the token at offset 2+j
        (hidden[:, :-o] predicts targets[:, o:]), decoded by the shared lm head;
        averaged over heads and scaled by mtp_weight."""
        mtp = hidden.new_zeros(())
        for j, head in enumerate(self.model.mtp_heads):
            o = j + 2
            transformed = head(hidden[:, :-o])
            mtp = mtp + self._head_ce(
                transformed, self.model.lmhead.weight, targets[:, o:]
            )
        return self.mtp_weight * mtp / self.model.n_mtp

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


class GPT(LMTrainerMixin, L.LightningModule):
    def __init__(
        self,
        transformer_lm: TransformerLM,
        pad_idx: int = 0,
        bos_idx: int | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        optimizer: str = "muon",
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int | None = None,
        min_lr_ratio: float = 0.1,
        grad_clip: float | None = 1.0,
        accumulate: int = 1,
        compile: bool | None = None,
        fused_loss: bool = False,
        tokenizer=None,
        sample_batches: int = 20,
        model_config: dict | None = None,
        mtp_weight: float = 0.3,
    ):
        super().__init__()
        # Weight on the auxiliary multi-token-prediction heads' loss (see
        # LMTrainerMixin._next_token_loss); ignored when the model has no MTP heads.
        self.mtp_weight = mtp_weight
        # `model_config` is the plain-dict build_lm(**model_config) recipe used to
        # construct `transformer_lm` (size/vocab_size/max_seq_len/overrides).
        # Saving it (and nothing else -- transformer_lm/tokenizer aren't
        # cleanly picklable/yaml-able) lets a checkpoint's own
        # hyper_parameters rebuild the exact same architecture later, instead
        # of relying on the caller to pass matching flags by hand.
        self.save_hyperparameters("model_config")
        self.model = transformer_lm
        self.pad_idx = pad_idx
        # A packed pretraining row holds several <|begin_of_text|>doc
        # <|end_of_text|> documents plus a <|pad|> tail; when bos_idx is set,
        # _loss derives per-token document ids from the <|begin_of_text|>
        # markers so attention never crosses a document boundary
        # (MosaicBERT-style sequence packing). None -> plain causal attention.
        self.bos_idx = bos_idx
        # During validation, log a generated continuation for batches with
        # batch_idx <= sample_batches (decode is slow, so only the first few).
        self.sample_batches = sample_batches
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
            fused_loss=fused_loss,
        )
        # Manual optimization: mirrors SFTModule so the two share the same LR
        # schedule / gradient-accumulation step (see LMTrainerMixin._optimizer_step),
        # which Lightning's automatic loop can't express. We own the optimizer
        # step, gradient accumulation, clipping and LR schedule here.
        self.automatic_optimization = False
        self._setup_compiled_forward(compile)

    def _loss(self, x: Tensor) -> Tensor:
        # Next-token prediction: position i's logits predict token i+1, so
        # compare the logits against the input shifted left by one.
        masks = None
        if self.bos_idx is not None:
            # Every <|begin_of_text|> starts a new document, so a running count of them
            # numbers the documents packed into this row. The attention masks
            # are built here, outside the compiled forward, and passed in as
            # inputs -- see Transformer.packed_masks.
            doc_ids = (x == self.bos_idx).cumsum(-1)
            # Rows packed by base_setup.py end in a <|pad|> tail; count pads
            # too so no pad position attends into the last document (their
            # targets are already ignore_index'd below).
            doc_ids = doc_ids + (x == self.pad_idx).cumsum(-1)
            masks = self.model.transformer.packed_masks(doc_ids)
        # Pretraining targets are the input itself (shifted inside the helper).
        return self._next_token_loss(x, x, masks)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._loss(batch)
        self._backward_and_step(loss, batch_idx)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._loss(batch)
        # sync_dist: under DDP each rank sees its own shard of the val set, so
        # the checkpoint monitor would otherwise rank on rank 0's local mean.
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        if batch_idx <= self.sample_batches:
            # Sanity-check what the model actually generates: prefill the first
            # half of the sequence and let it autoregress the second half, then
            # log prompt/generated/reference side by side to TensorBoard.
            self._log_generation_sample(batch, batch_idx)
        return loss

    def _log_generation_sample(self, batch: Tensor, batch_idx: int) -> None:
        # Rank 0 only: on other ranks the logger's experiment is a
        # DummyExperiment whose no-op add_text still passes the hasattr check
        # below, so without this guard every rank would pay for the slow
        # greedy decode just to throw the text away. Safe to return early --
        # nothing below involves a collective (decode runs in eval mode, so
        # the MoE bias all-reduce is not hit).
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and not trainer.is_global_zero:
            return
        # Need a tokenizer to render text and a TensorBoard writer to log it.
        writer = getattr(self.logger, "experiment", None)
        if self.tokenizer is None or writer is None or not hasattr(writer, "add_text"):
            return
        seq = batch[0]  # one example per logged batch is enough
        half = seq.shape[0] // 2
        if half == 0:
            return
        prompt, reference = seq[:half], seq[half:]
        generated = self._generate(prompt[None], max_new_tokens=reference.shape[0])[0]
        text = (
            f"**prompt**\n\n{self._decode_text(prompt)}\n\n"
            f"**generated**\n\n{self._decode_text(generated)}\n\n"
            f"**reference**\n\n{self._decode_text(reference)}"
        )
        writer.add_text(f"val_sample/{batch_idx}", text, self.global_step)


class SFTModule(LMTrainerMixin, L.LightningModule):
    """SFT (supervised fine-tuning) LightningModule.

    Kept separate from GPT (pretraining) rather than folded into it: the two
    build their loss from different batch shapes (see the module docstring);
    optimizer/LR-schedule/generation code is shared via LMTrainerMixin.
    """

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
        muon_weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int | None = None,
        min_lr_ratio: float = 0.1,
        grad_clip: float | None = 1.0,
        accumulate: int = 1,
        compile: bool | None = None,
        fused_loss: bool = False,
        tokenizer=None,
        model_config: dict | None = None,
        mtp_weight: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters("model_config")
        self.model = transformer_lm
        self.pad_idx = pad_idx
        # Weight on the auxiliary multi-token-prediction loss (see
        # LMTrainerMixin._next_token_loss); ignored without MTP heads.
        self.mtp_weight = mtp_weight
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
            fused_loss=fused_loss,
        )
        # Manual optimization: mirrors GPT so the two share the same LR
        # schedule / gradient-accumulation step (see LMTrainerMixin._optimizer_step).
        self.automatic_optimization = False
        self._setup_compiled_forward(compile)

    def _loss(
        self, input_ids: Tensor, labels: Tensor, doc_ids: Tensor | None = None
    ) -> Tensor:
        # Next-token prediction: position i's logits predict token i+1, so
        # compare against labels shifted left by one -- labels is already
        # input-aligned (see picochat.tokenizer.encode_conversation), not
        # pre-shifted. doc_ids marks which packed conversation each token
        # belongs to (see picochat.dataloader.pack_examples) so attention
        # stays within one conversation; its masks are built here, outside
        # the compiled forward -- see Transformer.packed_masks.
        masks = None
        if doc_ids is not None:
            masks = self.model.transformer.packed_masks(doc_ids)
        return self._next_token_loss(input_ids, labels, masks)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._loss(batch["input_ids"], batch["labels"], batch.get("doc_ids"))
        self._backward_and_step(loss, batch_idx)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._loss(batch["input_ids"], batch["labels"], batch.get("doc_ids"))
        # sync_dist: see GPT.validation_step -- the checkpoint monitor should
        # rank on the all-rank mean, not rank 0's shard.
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss


def _model_config_from_ckpt(ckpt, checkpoint: str) -> dict:
    """Pull and validate the saved `model_config` (the build_lm recipe
    GPT.__init__ stores) from a loaded Lightning checkpoint."""
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"{checkpoint} doesn't look like a Lightning checkpoint")
    model_config = (ckpt.get("hyper_parameters") or {}).get("model_config")
    if model_config is None:
        raise ValueError(
            f"{checkpoint} has no 'model_config' hyperparameter -- it predates "
            "GPT.__init__ saving it, so its architecture can't be rebuilt. "
            "Retrain to produce a checkpoint with model_config."
        )
    return model_config


def load_lm_from_checkpoint(
    checkpoint: str,
    vocab_size: int,
    overrides: dict | None = None,
    ckpt=None,
) -> tuple[TransformerLM, dict]:
    """Rebuild a bare TransformerLM from a checkpoint's saved `model_config`,
    apply `overrides` (e.g. max_seq_len for continual learning), load
    its weights (stripping GPT's `model.` state_dict prefix), and return
    (lm, model_config). Pass an already-loaded `ckpt` dict when several models
    come from one file (GRPO's policy + reference). Used by sft_train/grpo_train;
    load_gpt_checkpoint is the GPT-wrapping variant for inference CLIs."""
    if ckpt is None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = {**_model_config_from_ckpt(ckpt, checkpoint), **(overrides or {})}
    lm = build_lm(**{**model_config, "vocab_size": vocab_size})
    # GPT's state_dict keys are "model.*" (the wrapped TransformerLM) plus the
    # trainer scaffolding around it; strip the prefix to load into a bare lm.
    prefix = "model."
    state = {
        k[len(prefix) :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith(prefix)
    }
    lm.load_state_dict(state)
    return lm, model_config


def load_gpt_checkpoint(
    checkpoint: str, tokenizer_path: str, device: torch.device | str = "cpu"
) -> tuple[GPT, tiktoken.Encoding]:
    """Load a GPT + tokenizer for inference from a Lightning checkpoint.

    The architecture is rebuilt from the checkpoint's own `model_config`
    hyperparameter (the build_lm() recipe GPT.__init__ saves), so the caller
    never has to pass matching flags by hand. Used by scripts/chat.py
    and scripts/base_eval.py; requires a checkpoint produced by the current
    scripts/base_train.py or sft_train.py."""
    from picochat.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_path)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = _model_config_from_ckpt(ckpt, checkpoint)
    print(f"using model_config from checkpoint: {model_config}", flush=True)
    lm = build_lm(**{**model_config, "vocab_size": tokenizer.n_vocab})

    gpt = GPT(lm, compile=False, tokenizer=tokenizer, model_config=model_config)
    gpt.load_state_dict(ckpt["state_dict"])
    gpt.eval()
    gpt.to(device)
    return gpt, tokenizer
