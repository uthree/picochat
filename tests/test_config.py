"""The multi-device scaling helpers the training CLIs share (picochat.config):
device-count resolution from --devices/--accelerator, the DDP-only strategy
guard, and the linear-scaling rule over lr/max_steps/warmup_steps."""

import pytest

from picochat.config import check_strategy, resolve_num_devices, scale_for_devices


def test_resolve_num_devices_explicit_forms():
    assert resolve_num_devices("2") == 2
    assert resolve_num_devices("0,1") == 2
    assert resolve_num_devices("1") == 1


def test_resolve_num_devices_auto_on_cpu_accelerator_is_one():
    # --accelerator cpu --devices auto must not count visible CUDA devices:
    # Lightning's CPUAccelerator runs one process for 'auto'.
    assert resolve_num_devices("auto", accelerator="cpu") == 1


def test_check_strategy_accepts_ddp_variants():
    for s in ("auto", "ddp", "ddp_spawn", "ddp_find_unused_parameters_true"):
        assert check_strategy(s) == s


@pytest.mark.parametrize("strategy", ["fsdp", "FSDP", "deepspeed_stage_2"])
def test_check_strategy_rejects_sharded_strategies(strategy):
    # The trainers assume replicated parameters (grad clipping over
    # self.parameters(), DDP no_sync, MoE bias all-reduce); sharded
    # strategies must fail fast instead of training something subtly wrong.
    with pytest.raises(SystemExit):
        check_strategy(strategy)


def test_scale_for_devices_single_device_is_noop():
    lr, muon_lr, max_steps, warmup = scale_for_devices(
        {},
        1,
        lr_default=3e-4,
        muon_lr_default=0.02,
        max_steps_default=10000,
        warmup_steps_default=2000,
    )
    assert (lr, muon_lr, max_steps, warmup) == (3e-4, 0.02, 10000, 2000)


def test_scale_for_devices_applies_linear_scaling():
    lr, muon_lr, max_steps, warmup = scale_for_devices(
        {"lr": 1e-4, "muon_lr": 0.01, "max_steps": 8000, "warmup_steps": 400},
        4,
        lr_default=3e-4,
        muon_lr_default=0.02,
        max_steps_default=10000,
        warmup_steps_default=2000,
    )
    assert lr == pytest.approx(4e-4)
    assert muon_lr == pytest.approx(0.04)
    assert max_steps == 2000
    assert warmup == 100


def test_scale_for_devices_warmup_floor_and_zero():
    # A nonzero warmup never rounds down to 0 ...
    *_, warmup = scale_for_devices(
        {"warmup_steps": 1},
        8,
        lr_default=3e-4,
        muon_lr_default=0.02,
        max_steps_default=100,
        warmup_steps_default=2000,
    )
    assert warmup == 1
    # ... while an explicit warmup of 0 stays 0.
    *_, warmup = scale_for_devices(
        {"warmup_steps": 0},
        8,
        lr_default=3e-4,
        muon_lr_default=0.02,
        max_steps_default=100,
        warmup_steps_default=2000,
    )
    assert warmup == 0
