"""Training-side LightningModules for the model in gpt.py: GPT (pretraining)
and SFTModule (supervised fine-tuning), plus the LMTrainerMixin scaffolding
they share -- Muon/AdamW optimizer wiring, the warmup+cosine LR schedule
applied by hand under manual optimization, and greedy KV-cache generation.
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

import lightning as L
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from picochat.gpt import TransformerLM, build_lm, moe_modules, set_moe_top_k
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
        stochastic_k: tuple[int, int] | None = None,
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
        # Stochastic-k (MoE): if set to (min_k, max_k), each training step samples
        # one router top-k in that inclusive range and applies it to every MoE
        # layer, so the model learns to work across a range of active-expert
        # counts and top-k becomes a real quality/compute dial at inference (set
        # it with picochat.gpt.set_moe_top_k). Validation and inference use the
        # preset's nominal top-k (captured here per layer). None disables it (and
        # it is a no-op on a dense model).
        self.stochastic_k = tuple(stochastic_k) if stochastic_k is not None else None
        self._nominal_top_k = [m.n_active for m in moe_modules(self.model)]

    def _restore_nominal_top_k(self) -> None:
        for m, k in zip(moe_modules(self.model), self._nominal_top_k):
            m.n_active = k

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        # Sample one top-k for the whole model this step (a single value across
        # layers -- matches how inference sets top-k, and keeps every layer's
        # forward consistent). Uses torch so it honours Lightning's seeding.
        if self.stochastic_k is None or not self._nominal_top_k:
            return
        lo, hi = self.stochastic_k
        k = int(torch.randint(lo, hi + 1, (1,)).item())
        set_moe_top_k(self.model, k)

    def on_validation_start(self) -> None:
        # Always validate at the nominal top-k so val_loss stays comparable
        # across steps regardless of the k sampled for the surrounding batch.
        if self.stochastic_k is not None:
            self._restore_nominal_top_k()

    def on_fit_end(self) -> None:
        # Leave the model at its nominal top-k (checkpoints rebuild from the
        # preset's nominal n_active anyway, but keep the live object consistent).
        if self.stochastic_k is not None:
            self._restore_nominal_top_k()

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

    def _next_token_loss(self, input_ids: Tensor, targets: Tensor, masks) -> Tensor:
        """Next-token cross-entropy: position i's prediction is scored against
        targets[i+1] (`targets` is input-aligned, not pre-shifted; it is
        simply `input_ids` for pretraining, the masked labels for SFT).
        Positions whose target is pad_idx are ignored.

        With fused_loss, `self._forward` is the model trunk up to the final
        hidden states (TransformerLM.encode) and the lm-head matmul is folded
        into the loss kernel so the (b*l, vocab) logits are never
        materialized (see picochat.kernels.fused_linear_cross_entropy);
        otherwise `self._forward` is the full model and this is a plain
        F.cross_entropy over its logits.
        """
        if self.fused_loss:
            hidden = self._forward(input_ids, masks=masks)[:, :-1]
            if hidden.is_cuda:
                return fused_linear_cross_entropy(
                    hidden.reshape(-1, hidden.shape[-1]),
                    self.model.lmhead.weight,
                    targets[:, 1:].reshape(-1),
                    ignore_index=self.pad_idx,
                )
            # The kernel is available in this process but the module runs on
            # CPU (e.g. a CPU Trainer in tests): apply the lm-head eagerly and
            # fall through to the plain loss on its logits.
            logits = self.model.lmhead(hidden)
        else:
            logits = self._forward(input_ids, masks=masks)[:, :-1]
        return F.cross_entropy(
            rearrange(logits, "b l v -> (b l) v"),
            rearrange(targets[:, 1:], "b l -> (b l)"),
            ignore_index=self.pad_idx,
        )

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
        stochastic_k: tuple[int, int] | None = None,
    ):
        super().__init__()
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
            stochastic_k=stochastic_k,
        )
        # Manual optimization: mirrors SFTModule so the two share the same LR
        # schedule / gradient-accumulation step (see LMTrainerMixin._optimizer_step),
        # which Lightning's automatic loop can't express. We own the optimizer
        # step, gradient accumulation, clipping and LR schedule here.
        self.automatic_optimization = False
        # `compile=None` -> auto (compile iff the environment supports it). The
        # compiled handle shares parameters with self.model, which stays
        # uncompiled, so decode() runs eager. With fused_loss the trainable
        # forward stops at the hidden states -- the lm-head runs inside the
        # loss kernel instead (see _next_token_loss). Stored via
        # object.__setattr__ so nn.Module.__setattr__ doesn't register it as
        # a submodule: registered, it would duplicate every weight in
        # state_dict under `_forward.*` (or `_forward._orig_mod.*` when
        # compiled), making checkpoints loadable only under the exact same
        # compile setting.
        self.compile = can_compile() if compile is None else compile
        trunk = self.model.encode if self.fused_loss else self.model
        object.__setattr__(
            self, "_forward", torch.compile(trunk) if self.compile else trunk
        )

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
        (loss / self.accumulate).backward()
        self._optimizer_step(batch_idx)
        self.log("train_loss", loss)
        self.log("loss", loss, prog_bar=True, logger=False)  # for progress bar
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        if batch_idx <= self.sample_batches:
            # Sanity-check what the model actually generates: prefill the first
            # half of the sequence and let it autoregress the second half, then
            # log prompt/generated/reference side by side to TensorBoard.
            self._log_generation_sample(batch, batch_idx)
        return loss

    def _log_generation_sample(self, batch: Tensor, batch_idx: int) -> None:
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
        stochastic_k: tuple[int, int] | None = None,
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
            muon_weight_decay=muon_weight_decay,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
            grad_clip=grad_clip,
            accumulate=accumulate,
            tokenizer=tokenizer,
            fused_loss=fused_loss,
            stochastic_k=stochastic_k,
        )
        # Manual optimization: mirrors GPT so the two share the same LR
        # schedule / gradient-accumulation step (see LMTrainerMixin._optimizer_step).
        self.automatic_optimization = False
        # object.__setattr__: keep the forward handle out of _modules so it
        # doesn't duplicate weights in state_dict (see GPT.__init__, including
        # the fused_loss trunk choice).
        self.compile = can_compile() if compile is None else compile
        trunk = self.model.encode if self.fused_loss else self.model
        object.__setattr__(
            self, "_forward", torch.compile(trunk) if self.compile else trunk
        )

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
        (loss / self.accumulate).backward()
        self._optimizer_step(batch_idx)
        self.log("train_loss", loss)
        self.log("loss", loss, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._loss(batch["input_ids"], batch["labels"], batch.get("doc_ids"))
        self.log("val_loss", loss, prog_bar=True)
        return loss


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
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"{checkpoint} doesn't look like a Lightning checkpoint")
    model_config = (ckpt.get("hyper_parameters") or {}).get("model_config")
    if model_config is None:
        raise ValueError(
            f"{checkpoint} has no 'model_config' hyperparameter -- it predates "
            "GPT.__init__ saving it, so its architecture can't be rebuilt. "
            "Retrain to produce a checkpoint with model_config."
        )

    print(f"using model_config from checkpoint: {model_config}", flush=True)
    lm = build_lm(**{**model_config, "vocab_size": tokenizer.n_vocab})

    gpt = GPT(lm, compile=False, tokenizer=tokenizer, model_config=model_config)
    gpt.load_state_dict(ckpt["state_dict"])
    gpt.eval()
    gpt.to(device)
    return gpt, tokenizer
