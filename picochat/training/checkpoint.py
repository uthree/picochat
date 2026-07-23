"""Loading trained checkpoints back into models: the bare-TransformerLM
loader the training CLIs use to warm-start (SFT/GRPO) and the GPT-wrapping
variant the inference CLIs use."""

import torch

from picochat.model.transformer import TransformerLM
from picochat.model.presets import build_lm
from picochat.tokenizer import Tokenizer, load_tokenizer
from picochat.training.modules import GPT


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


def load_mm_encoders(ckpt: dict, d_model: int):
    """Rebuild the media encoders (audio, vision) a multimodal SFT checkpoint
    carries: architecture from the saved `mm_config` hyperparameter, weights
    from the checkpoint's `audio_encoder.*` / `vision_encoder.*` keys -- no
    Hub access needed at serve time. Returns (None, None) for a text-only
    checkpoint."""
    from picochat.model.multimodal import rebuild_encoders

    mm_config = (ckpt.get("hyper_parameters") or {}).get("mm_config")
    audio_encoder, vision_encoder = rebuild_encoders(mm_config, d_model)
    for name, enc in (
        ("audio_encoder", audio_encoder),
        ("vision_encoder", vision_encoder),
    ):
        if enc is None:
            continue
        prefix = name + "."
        state = {
            k[len(prefix) :]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith(prefix)
        }
        enc.load_state_dict(state)
        enc.eval()
    return audio_encoder, vision_encoder


def load_gpt_checkpoint(
    checkpoint: str,
    tokenizer_path: str,
    device: torch.device | str = "cpu",
    ckpt=None,
) -> tuple[GPT, Tokenizer]:
    """Load a GPT + tokenizer for inference from a Lightning checkpoint.

    The architecture is rebuilt from the checkpoint's own `model_config`
    hyperparameter (the build_lm() recipe GPT.__init__ saves), so the caller
    never has to pass matching flags by hand. Used by scripts/chat.py
    and scripts/base_eval.py; requires a checkpoint produced by the current
    scripts/base_train.py or sft_train.py. Pass an already-loaded `ckpt` dict
    to avoid re-reading the file (e.g. scripts/api.py, which also pulls the
    media encoders out of the same checkpoint)."""
    tokenizer = load_tokenizer(tokenizer_path)

    if ckpt is None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = _model_config_from_ckpt(ckpt, checkpoint)
    print(f"using model_config from checkpoint: {model_config}", flush=True)
    lm = build_lm(**{**model_config, "vocab_size": tokenizer.n_vocab})

    gpt = GPT(lm, compile=False, tokenizer=tokenizer, model_config=model_config)
    # A multimodal SFT checkpoint also carries the media encoders' weights;
    # they are not part of the language model (load_mm_encoders rebuilds them
    # separately), so drop them here to keep this load strict.
    state = {
        k: v
        for k, v in ckpt["state_dict"].items()
        if not k.startswith(("audio_encoder.", "vision_encoder."))
    }
    gpt.load_state_dict(state)
    gpt.eval()
    gpt.to(device)
    return gpt, tokenizer
