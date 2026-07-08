import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from picochat.model.lm_module import LMTrainerMixin, can_compile
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
        tie_embeddings=True,
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
        tie_embeddings=True,
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
        n_heads=16,
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


class GPT(LMTrainerMixin, L.LightningModule):
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
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
            grad_clip=grad_clip,
            accumulate=accumulate,
            tokenizer=tokenizer,
        )
        # Manual optimization: training backprops each lm head separately off a
        # single trunk forward (see _mtp_backward / training_step), which the
        # automatic loop -- one loss.backward() -- can't express. We therefore
        # own the optimizer step, gradient accumulation, clipping and LR
        # schedule here instead of leaving them to the Trainer.
        self.automatic_optimization = False
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
