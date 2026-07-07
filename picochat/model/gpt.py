import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from picochat.model.transformer import (
    MixtureOfExperts,
    SelfAttention,
    SwiGLU,
    Transformer,
    TransformerLayer,
    TransformerLM,
    estimate_num_params,
    rms_norm,
    rotate_half,
)
from picochat.optim import Muon

# Re-exported so `from picochat.model.gpt import ...` keeps working after the
# architecture moved to transformer.py.
__all__ = [
    "MixtureOfExperts",
    "SelfAttention",
    "SwiGLU",
    "Transformer",
    "TransformerLayer",
    "TransformerLM",
    "estimate_num_params",
    "rms_norm",
    "rotate_half",
    "MODEL_PRESETS",
    "build_lm",
    "can_compile",
    "estimate_preset_params",
    "load_state_dict_expand",
    "GPT",
]


# Scale ladder.
MODEL_PRESETS: dict[str, dict] = {
    "pico": dict(
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=64000,
        window_size=128,
        global_attn_ratio=4,
        n_lmheads=1,
    ),
    "small": dict(
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=2,
        vocab_size=64000,
        window_size=128,
        global_attn_ratio=4,
        n_lmheads=1,
    ),
    "base": dict(
        d_model=1024,
        n_layers=18,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=64000,
        window_size=128,
        global_attn_ratio=6,
        n_lmheads=4,
    ),
    "medium": dict(
        d_model=2048,
        n_layers=24,
        n_heads=12,
        n_kv_heads=4,
        vocab_size=64000,
        window_size=256,
        global_attn_ratio=6,
        n_experts=64,
        d_expert=1024,
        n_lmheads=4,
    ),
    "large": dict(
        d_model=2560,
        n_layers=30,
        n_heads=20,
        n_kv_heads=4,
        vocab_size=64000,
        window_size=256,
        global_attn_ratio=6,
        n_experts=256,
        d_expert=1024,
        n_lmheads=4,
    ),
}


def build_lm(
    size: str,
    vocab_size: int | None = None,
    max_seq_len: int = 4096,
    **overrides,
) -> TransformerLM:
    """Build a TransformerLM from a preset name. vocab_size defaults to the
    preset's recommended value; pass it explicitly (e.g. the tokenizer's actual
    vocab) to override. Any other field can be overridden via overrides."""
    if size not in MODEL_PRESETS:
        raise ValueError(f"unknown size '{size}'. choices: {list(MODEL_PRESETS)}")
    cfg = {**MODEL_PRESETS[size], **overrides}
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return TransformerLM(max_seq_len=max_seq_len, **cfg)


def estimate_preset_params(
    size: str,
    vocab_size: int | None = None,
    active_only: bool = False,
    **overrides,
) -> int:
    """Estimate the parameter count of build_lm(size, ...) without building it.

    Same preset/override resolution as build_lm, so the two always describe the
    same model. Handy for sizing the scale ladder on a machine that can't hold
    the larger presets in memory. active_only=True returns the per-token active
    parameter count instead of the total (see estimate_num_params)."""
    if size not in MODEL_PRESETS:
        raise ValueError(f"unknown size '{size}'. choices: {list(MODEL_PRESETS)}")
    cfg = {**MODEL_PRESETS[size], **overrides}
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return estimate_num_params(**cfg, active_only=active_only)


def can_compile() -> bool:
    """Whether torch.compile is likely to help in this environment.

    The inductor backend targets CUDA; on CPU/MPS it often falls back or errors,
    so we only enable it on CUDA. torch.compile itself is lazy (compiles on the
    first forward), so this just gates whether we wrap the model at all.
    """
    return hasattr(torch, "compile") and torch.cuda.is_available()


def load_state_dict_expand(module: nn.Module, state_dict: dict) -> dict[str, int]:
    """Load a checkpoint into `module`, expanding tensors whose shape grew.

    Used to warm-start a bigger config (e.g. larger d_model/n_layers) from a
    smaller checkpoint. For each tensor in `module`'s own state_dict:
      - key missing from `state_dict`: left as-is (module's own random init).
      - same shape: copied verbatim.
      - same ndim, different shape: the checkpoint tensor is copied into the
        low-index corner (e.g. top-left for a matrix) up to
        min(old_size, new_size) along every axis; the rest of the module's
        random init is kept untouched. This also covers shrinking (a larger
        checkpoint loaded into a smaller module).
      - different ndim: left as-is (skipped, shapes are incompatible).
    Returns counts of tensors handled each way, keyed by "matched"/"expanded"/
    "skipped".
    """
    own_state = module.state_dict()
    stats = {"matched": 0, "expanded": 0, "skipped": 0}
    with torch.no_grad():
        for key, own_tensor in own_state.items():
            if key not in state_dict:
                stats["skipped"] += 1
                continue
            src = state_dict[key]
            if src.shape == own_tensor.shape:
                own_tensor.copy_(src)
                stats["matched"] += 1
            elif src.dim() == own_tensor.dim():
                region = tuple(
                    slice(0, min(s, d)) for s, d in zip(src.shape, own_tensor.shape)
                )
                own_tensor[region].copy_(src[region])
                stats["expanded"] += 1
            else:
                stats["skipped"] += 1
    return stats


class GPT(L.LightningModule):
    def __init__(
        self,
        transformer_lm: TransformerLM,
        pad_idx: int = 0,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        optimizer: str = "muon",
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        warmup_steps: int = 2000,
        max_steps: int | None = None,
        min_lr_ratio: float = 0.1,
        grad_clip: float | None = 1.0,
        accumulate: int = 1,
        compile: bool | None = None,
        tokenizer=None,
        sample_batches: int = 20,
        model_config: dict | None = None,
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
        # Optional tiktoken Encoding used to turn generated token ids back into
        # readable text for the TensorBoard generation samples (see below).
        self.tokenizer = tokenizer
        # During validation, log a generated continuation for batches with
        # batch_idx <= sample_batches (decode is slow, so only the first few).
        self.sample_batches = sample_batches
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
        # Manual optimization: training backprops each lm head separately off a
        # single trunk forward (see _mtp_backward / training_step), which the
        # automatic loop -- one loss.backward() -- can't express. We therefore
        # own the optimizer step, gradient accumulation, clipping and LR
        # schedule here instead of leaving them to the Trainer.
        self.automatic_optimization = False
        # Set by configure_optimizers: the per-group base LR the schedule scales.
        self._base_lrs: list[float] = []
        # `compile=None` -> auto (compile iff the environment supports it). We
        # compile only the shared trunk (encode); the lm heads are single
        # matmuls left eager so _mtp_backward can drive them one at a time. The
        # compiled handle shares parameters with self.model; stashing it in a
        # list keeps nn.Module from registering it as a submodule (which would
        # duplicate every parameter under a `_orig_mod.` prefix and break
        # checkpoint loading). self.model stays uncompiled, so state_dict keys
        # stay clean and decode() runs eager.
        self.compile = can_compile() if compile is None else compile
        self._encode = [
            torch.compile(self.model.encode) if self.compile else self.model.encode
        ]

    def _head_loss(self, h: Tensor, x: Tensor, k: int) -> Tensor:
        # Multiple token prediction: head k's output at position i predicts token
        # i+1+k, so each head shifts the targets one step further (head 0 is
        # ordinary next-token prediction). `h` is the shared trunk hidden state.
        shift = k + 1
        logits = rearrange(self.model.lmheads[k](h)[:, :-shift], "b l v -> (b l) v")
        targets = rearrange(x[:, shift:], "b l -> (b l)")
        return F.cross_entropy(logits, targets, ignore_index=self.pad_idx)

    def _head_losses(self, x: Tensor) -> Tensor:
        # Per-head losses in one shot (no backward). Used for validation and as a
        # reference in tests; the memory-optimized training path is
        # _mtp_backward. One head's (B, L, V) logits is built at a time, so peak
        # logits memory is independent of the head count.
        h = self._encode[0](x)
        return torch.stack(
            [self._head_loss(h, x, k) for k in range(len(self.model.lmheads))]
        )

    def _loss(self, x: Tensor) -> Tensor:
        return self._head_losses(x).mean()

    def _mtp_backward(self, x: Tensor, scale: float) -> Tensor:
        """Forward the shared trunk once, then backprop each lm head separately
        so only one head's (B, L, V) logits is ever materialized for backward --
        peak activation memory stays flat in the head count instead of holding
        all heads at once. `scale` folds in gradient accumulation (1/accumulate).
        Returns the detached per-head losses (unscaled) for logging.

        Assumes bf16/fp32 (no fp16 GradScaler): plain .backward() is used so the
        two-stage backward composes correctly and the method also works when
        called outside a Trainer.
        """
        n = len(self.model.lmheads)
        h = self._encode[0](x)  # (B, L, d); graph runs back to the trunk params
        # Detach the trunk so each head backprops through its own small graph and
        # frees its logits immediately; the gradients w.r.t. the hidden state
        # accumulate into h_leaf.grad across heads.
        h_leaf = h.detach().requires_grad_(True)
        head_losses = []
        for k in range(n):
            loss_k = self._head_loss(h_leaf, x, k)
            # 1/n: mean over heads (matches _loss). scale: gradient accumulation.
            (loss_k * (scale / n)).backward()
            head_losses.append(loss_k.detach())
        # One backward through the trunk, driven by the head gradients collected
        # in h_leaf.grad (which already carries the scale/n factor). Equivalent to
        # backprop of sum_k loss_k*scale/n, but with the heads done sequentially.
        h.backward(h_leaf.grad)
        return torch.stack(head_losses)

    def _log_head_losses(self, prefix: str, head_losses: Tensor) -> None:
        # Per-head breakdown (head 0 is the loss comparable to a single-head
        # run); skip when there's nothing to break down.
        if head_losses.numel() > 1:
            for k, head_loss in enumerate(head_losses):
                self.log(f"{prefix}_head{k}", head_loss)

    def _optimizer_step(self, batch_idx: int) -> None:
        # Manual optimization: step once every `accumulate` microbatches, applying
        # the LR schedule and gradient clipping ourselves. No-op when not attached
        # to a Trainer (e.g. training_step called directly in a unit test) --
        # _mtp_backward has already populated .grad there.
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

    def _apply_lr(self, opt) -> None:
        # Scale each param group's base LR by the warmup/cosine schedule. Keyed on
        # global_step (optimizer steps), matching the old "interval: step"
        # scheduler. When max_steps is unknown, keep LR constant (as the previous
        # optimizer-only path did).
        if self.max_steps is None:
            return
        scale = self._lr_lambda(self.trainer.global_step)
        for base, group in zip(self._base_lrs, opt.param_groups):
            group["lr"] = base * scale

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        head_losses = self._mtp_backward(batch, scale=1.0 / self.accumulate)
        loss = head_losses.mean()
        self._optimizer_step(batch_idx)
        self.log("train_loss", loss)
        self._log_head_losses("train_loss", head_losses)
        self.log("loss", loss, prog_bar=True, logger=False)  # for progress bar
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        head_losses = self._head_losses(batch)
        loss = head_losses.mean()
        self.log("val_loss", loss, prog_bar=True)
        self._log_head_losses("val_loss", head_losses)
        if batch_idx <= self.sample_batches:
            # Sanity-check what the model actually generates: prefill the first
            # half of the sequence and let it autoregress the second half, then
            # log prompt/generated/reference side by side to TensorBoard.
            self._log_generation_sample(batch, batch_idx)
        return loss

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
