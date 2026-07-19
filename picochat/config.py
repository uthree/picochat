"""Shared config-loading and multi-device scaling helpers for the training CLIs.

Small glue used across scripts/ (base_train, sft_train, grpo_train, ...): read a
YAML recipe, work out the world size the --devices/--num-nodes flags imply,
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


def resolve_num_devices(devices: str, accelerator: str = "auto") -> int:
    """Per-node device count implied by --devices, for scaling lr/max_steps.

    Mirrors how Lightning itself interprets the flag ('auto' -> all visible
    GPUs, 'N' -> N devices, '0,1' -> that many device ids) without needing a
    Trainer instance up front. With --accelerator cpu, 'auto' means one
    process (matching Lightning's CPUAccelerator) even when CUDA devices are
    visible."""
    if devices == "auto":
        if accelerator == "cpu" or not torch.cuda.is_available():
            return 1
        return torch.cuda.device_count()
    if "," in devices:
        return len(devices.split(","))
    return int(devices)


def check_strategy(strategy: str) -> str:
    """Reject strategies that shard parameters/gradients (FSDP, DeepSpeed).

    The training modules are DDP-only: manual optimization clips gradients
    with torch.nn.utils.clip_grad_norm_ over replicated parameters, gradient
    accumulation suppresses sync via DDP's no_sync() (see
    LMTrainerMixin._grad_sync_context), and the MoE load-balancing bias
    all-reduce assumes every rank holds the full buffer. None of those hold
    once parameters are sharded, so fail fast instead of training something
    subtly wrong."""
    if any(s in strategy.lower() for s in ("fsdp", "deepspeed")):
        raise SystemExit(
            f"strategy '{strategy}' is not supported: the trainers assume "
            "replicated parameters (see picochat.config.check_strategy). "
            "Use 'auto', 'ddp' or another DDP variant."
        )
    return strategy


def scale_for_devices(
    optim_cfg: dict,
    world_size: int,
    *,
    lr_default: float,
    muon_lr_default: float,
    max_steps_default: int,
    warmup_steps_default: int,
) -> tuple[float, float, int, int]:
    """Apply the linear-scaling rule for a multi-GPU run and return
    (lr, muon_lr, max_steps, warmup_steps). lr scales up linearly with the
    larger effective batch; max_steps and warmup_steps scale down so the run
    still sees the same total tokens and the warmup keeps the same share of
    the schedule; a single device is a no-op. `world_size` is devices per node
    times number of nodes. Prints the adjustment when it changes anything."""
    base_lr = optim_cfg.get("lr", lr_default)
    base_muon_lr = optim_cfg.get("muon_lr", muon_lr_default)
    base_max_steps = optim_cfg.get("max_steps", max_steps_default)
    base_warmup = optim_cfg.get("warmup_steps", warmup_steps_default)
    lr = base_lr * world_size
    muon_lr = base_muon_lr * world_size
    max_steps = max(1, round(base_max_steps / world_size))
    warmup_steps = max(1, round(base_warmup / world_size)) if base_warmup else 0
    if world_size > 1:
        print(
            f"scaling for world size {world_size}: lr {base_lr} -> {lr}, "
            f"muon_lr {base_muon_lr} -> {muon_lr}, "
            f"max_steps {base_max_steps} -> {max_steps}, "
            f"warmup_steps {base_warmup} -> {warmup_steps}",
            flush=True,
        )
    return lr, muon_lr, max_steps, warmup_steps
