import lightning as L
import torch
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
    "GPT",
]


# Scale ladder. Sized for a 128k vocab and an untied lm head (embedding + head
# together are 256000*d_model params, which dominates the smaller presets).
# pico is the ~0.5B-param entry point; each step up roughly doubles the active
# parameter count. medium/large are sparse (MoE): total >> active.
MODEL_PRESETS: dict[str, dict] = {
    "pico": dict(  # ~0.5B params (dense)
        d_model=1024,
        n_layers=20,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=128000,
        window_size=128,
        global_attn_ratio=4,
    ),
    "small": dict(  # ~1.0B params (dense)
        d_model=1536,
        n_layers=24,
        n_heads=24,
        n_kv_heads=4,
        vocab_size=128000,
        window_size=128,
        global_attn_ratio=4,
    ),
    "base": dict(  # ~1.9B params (dense)
        d_model=2048,
        n_layers=28,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=128000,
        window_size=256,
        global_attn_ratio=6,
    ),
    "medium": dict(  # ~7.5B total / ~2.6B active (MoE)
        d_model=2048,
        n_layers=28,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=128000,
        window_size=256,
        global_attn_ratio=6,
        n_experts=32,
        n_active=4,
        d_expert=1024,
    ),
    "large": dict(  # ~23B total / ~4.9B active (MoE)
        d_model=2560,
        n_layers=32,
        n_heads=20,
        n_kv_heads=4,
        vocab_size=128000,
        window_size=512,
        global_attn_ratio=6,
        n_experts=64,
        n_active=6,
        d_expert=1280,
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
        # The packed pretraining stream is a run of <|begin_of_text|>doc
        # <|end_of_text|> documents; when bos_idx is set, _loss derives
        # per-token document ids from the <|begin_of_text|> markers so
        # attention never crosses a document boundary (MosaicBERT-style
        # sequence packing). None -> plain causal attention.
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
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
            grad_clip=grad_clip,
            accumulate=accumulate,
            tokenizer=tokenizer,
        )
        # Manual optimization: mirrors SFTModule so the two share the same LR
        # schedule / gradient-accumulation step (see LMTrainerMixin._optimizer_step),
        # which Lightning's automatic loop can't express. We own the optimizer
        # step, gradient accumulation, clipping and LR schedule here.
        self.automatic_optimization = False
        # `compile=None` -> auto (compile iff the environment supports it). The
        # compiled handle shares parameters with self.model, which stays
        # uncompiled, so decode() runs eager. Stored via object.__setattr__ so
        # nn.Module.__setattr__ doesn't register it as a submodule: registered,
        # it would duplicate every weight in state_dict under `_forward.*`
        # (or `_forward._orig_mod.*` when compiled), making checkpoints
        # loadable only under the exact same compile setting.
        self.compile = can_compile() if compile is None else compile
        object.__setattr__(
            self, "_forward", torch.compile(self.model) if self.compile else self.model
        )

    def _loss(self, x: Tensor) -> Tensor:
        # Next-token prediction: position i's logits predict token i+1, so
        # compare the logits against the input shifted left by one.
        masks = None
        if self.bos_idx is not None:
            # Every <|begin_of_text|> starts a new document, so a running count of them
            # numbers the documents packed into this window; tokens before the
            # first <|begin_of_text|> (a document cut mid-way by the window) form doc 0.
            # The attention masks are built here, outside the compiled
            # forward, and passed in as inputs -- see Transformer.packed_masks.
            doc_ids = (x == self.bos_idx).cumsum(-1)
            masks = self.model.transformer.packed_masks(doc_ids)
        logits = self._forward(x, masks=masks)[:, :-1]
        targets = x[:, 1:]
        return F.cross_entropy(
            rearrange(logits, "b l v -> (b l) v"),
            rearrange(targets, "b l -> (b l)"),
            ignore_index=self.pad_idx,
        )

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
