"""Shared config-loading and multi-device scaling helpers for the training CLIs.

Small glue used across scripts/ (base_train, sft_train, grpo_train, ...): read a
YAML recipe, work out the single-node device count a `--devices` flag implies,
and apply the linear-scaling rule that keeps a multi-GPU run equivalent to the
single-GPU dynamics the config was written for.
"""

from __future__ import annotations

import torch
import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_num_devices(devices: str) -> int:
    """Single-node device count implied by --devices, for scaling lr/max_steps.

    Mirrors how Lightning itself interprets the flag ('auto' -> all visible
    GPUs, 'N' -> N devices, '0,1' -> that many device ids) without needing a
    Trainer instance up front."""
    if devices == "auto":
        return torch.cuda.device_count() if torch.cuda.is_available() else 1
    if "," in devices:
        return len(devices.split(","))
    return int(devices)


def scale_for_devices(
    optim_cfg: dict,
    num_devices: int,
    *,
    lr_default: float,
    muon_lr_default: float,
    max_steps_default: int,
) -> tuple[float, float, int]:
    """Apply the linear-scaling rule for a multi-GPU run and return
    (lr, muon_lr, max_steps). lr scales up linearly with the larger effective
    batch and max_steps scales down so the run still sees the same total tokens;
    a single device is a no-op. Prints the adjustment when it changes anything."""
    base_lr = optim_cfg.get("lr", lr_default)
    base_muon_lr = optim_cfg.get("muon_lr", muon_lr_default)
    base_max_steps = optim_cfg.get("max_steps", max_steps_default)
    lr = base_lr * num_devices
    muon_lr = base_muon_lr * num_devices
    max_steps = max(1, round(base_max_steps / num_devices))
    if num_devices > 1:
        print(
            f"scaling for {num_devices} devices: lr {base_lr} -> {lr}, "
            f"muon_lr {base_muon_lr} -> {muon_lr}, "
            f"max_steps {base_max_steps} -> {max_steps}",
            flush=True,
        )
    return lr, muon_lr, max_steps
