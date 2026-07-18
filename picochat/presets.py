"""The model scale ladder and the factory helpers that size and build it.

Config-loading / factory glue kept out of gpt.py (which is just the model
definition): the {size: hyperparameters} presets live in configs/presets.yml so
they sit with the other recipes, and `build_lm` / `estimate_preset_params` turn
a preset name into a model / a parameter count. Re-exported from gpt.py for
back-compat, so `from picochat.gpt import build_lm, MODEL_PRESETS` still works.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from picochat.param_estimate import estimate_num_params

# Scale ladder: total-param rungs 200m/1b/8b/35b/120b, each crossed with a
# dense / -moe / -moe-shared architecture variant (dense up to 8b, MoE from 1b),
# kept in configs/presets.yml so the hyperparameters live with the other recipes
# (see that file for the naming and sizing rationale).
PRESETS_FILE = Path(__file__).resolve().parents[1] / "configs" / "presets.yml"


def load_presets(path: str | Path = PRESETS_FILE) -> dict[str, dict]:
    """Load the {size: hyperparameters} scale ladder consumed by build_lm."""
    with open(path) as f:
        return yaml.safe_load(f)


MODEL_PRESETS: dict[str, dict] = load_presets()


def _resolve_preset(
    size: str, vocab_size: int | None = None, **overrides
) -> dict:
    """Resolve a preset name + overrides into a model config dict. Shared by
    build_lm and estimate_preset_params so they always describe the same model."""
    if size not in MODEL_PRESETS:
        raise ValueError(f"unknown size '{size}'. choices: {list(MODEL_PRESETS)}")
    cfg = {**MODEL_PRESETS[size], **overrides}
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return cfg


def build_lm(
    size: str,
    vocab_size: int | None = None,
    max_seq_len: int = 4096,
    **overrides,
):
    """Build a TransformerLM from a preset name. vocab_size defaults to the
    preset's recommended value; pass it explicitly (e.g. the tokenizer's actual
    vocab) to override. Any other field can be overridden via overrides."""
    from picochat.gpt import TransformerLM  # lazy: avoid a gpt<->presets cycle

    cfg = _resolve_preset(size, vocab_size, **overrides)
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
    cfg = _resolve_preset(size, vocab_size, **overrides)
    return estimate_num_params(**cfg, active_only=active_only)
