"""Train one stage from a YAML recipe.

A stage = one training run defined entirely by a config file (model size, data
bins, optimizer, trainer). We do NOT orchestrate multiple stages here; to run a
curriculum, train one stage, then point the next stage's config at the produced
checkpoint via `init_from` to warm-start (continual learning) with a fresh
optimizer and LR schedule.

    python scripts/base_train.py --config configs/base_train/stage1_basic.yml

The model architecture must stay the same across stages that chain via
`init_from` (only the data / schedule change).
"""

import argparse
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import ConcatDataset

from picochat.data.pretrain import PackedDataset, PretrainDataModule
from picochat.model.gpt import GPT, build_lm, load_state_dict_expand
from picochat.tokenizer import load_tokenizer

# Fields under `model:` that override the scale-ladder preset.
MODEL_OVERRIDES = (
    "d_model",
    "n_heads",
    "n_kv_heads",
    "n_layers",
    "grad_checkpoint",
    "window_size",
    "n_lmheads",
    "tie_embeddings",
    "rope_base",
)


def resolve_num_devices(devices: str) -> int:
    """Single-node device count implied by --devices, for scaling lr/max_steps.

    Mirrors how Lightning itself would interpret the flag ('auto' -> all
    visible GPUs, 'N' -> N devices, '0,1' -> that many device ids) without
    needing a Trainer instance up front.
    """
    if devices == "auto":
        return torch.cuda.device_count() if torch.cuda.is_available() else 1
    if "," in devices:
        return len(devices.split(","))
    return int(devices)


def resolve_bins(bins, data_dir: str) -> list[str]:
    """Join `data_dir` with each bin filename (mirrors base_setup.py's own
    output_dir/output split, so both configs name files the same way)."""
    if isinstance(bins, str):
        bins = [bins]
    return [str(Path(data_dir) / b) for b in bins]


def resolve_datasets(datasets, data_dir: str) -> tuple[list[str], list[float]]:
    """Split `data.datasets` (list of {path, weight}) into parallel bin paths
    and weights, joining `data_dir` with each path."""
    paths = [str(Path(data_dir) / d["path"]) for d in datasets]
    weights = [d.get("weight", 1.0) for d in datasets]
    return paths, weights


def make_dataset(bins, block_size: int, random: bool, weights=None):
    """Build a (Concat)PackedDataset from a single path or a list of paths.

    Returns (dataset, group_weights). group_weights is None unless `weights`
    is given, in which case it's passed through unchanged: one weight per
    source dataset, consumed by GroupWeightedIndexSampler so each source's
    total sampling mass equals its configured weight regardless of its
    example count.
    """
    if isinstance(bins, str):
        bins = [bins]
    parts = [PackedDataset(b, block_size=block_size, random=random) for b in bins]
    if len(parts) == 1:
        if weights is not None:
            raise ValueError("train_weights requires more than one train_bin entry")
        return parts[0], None
    dataset = ConcatDataset(parts)
    if weights is None:
        return dataset, None
    if len(weights) != len(parts):
        raise ValueError(
            f"train_weights has {len(weights)} entries but train_bin has "
            f"{len(parts)}"
        )
    return dataset, list(weights)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="stage recipe (YAML)")
    p.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="checkpoint to warm-start from (overrides config's init_from)",
    )
    p.add_argument(
        "--init-expand",
        action="store_true",
        help=(
            "when init_from's model is smaller than the configured size, "
            "expand its weights into the larger tensors (low-index corner) "
            "instead of requiring an exact shape match (also settable via "
            "config's init_expand)"
        ),
    )
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto", help="e.g. 'auto', '2', or '0,1'")
    p.add_argument("--strategy", type=str, default="auto", help="e.g. 'auto', 'ddp', 'fsdp'")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = dict(cfg.get("model", {}))
    data_cfg = cfg.get("data", {})
    optim_cfg = cfg.get("optim", {})
    trainer_cfg = cfg.get("trainer", {})
    tokenizer_path = cfg.get("tokenizer", "weights/tokenizer.json")
    output_dir = cfg.get("output_dir", "weights")
    init_from = args.init_from or cfg.get("init_from")
    init_expand = args.init_expand or cfg.get("init_expand", False)

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.n_vocab
    pad_idx = tokenizer.encode_single_token("<pad>")

    # --- data ---
    block_size = data_cfg.get("block_size", 1024)
    # max_seq_len/rope_base (the latter via MODEL_OVERRIDES) can both be raised
    # in a later stage's config to extend context length via continual learning
    # (init_from): RoPE's sin/cos tables are non-persistent buffers rebuilt from
    # these at construction time, not part of the checkpoint, so changing them
    # doesn't affect any learned tensor's shape and init_from's plain
    # load_state_dict (no init_expand needed) still applies cleanly.
    max_seq_len = model_cfg.pop("max_seq_len", 4096)
    assert block_size < max_seq_len, "block_size+1 <= max_seq_len required"
    batch_size = trainer_cfg.get("batch_size", 2)
    num_workers = trainer_cfg.get("num_workers", 4)

    data_dir = data_cfg.get("data_dir", "data")
    train_bins, train_weights = resolve_datasets(data_cfg["datasets"], data_dir)
    train_ds, train_group_weights = make_dataset(
        train_bins,
        block_size,
        random=True,
        weights=train_weights if len(train_bins) > 1 else None,
    )
    val_ds = None
    monitor = None
    if data_cfg.get("val_bin"):
        val_ds, _ = make_dataset(resolve_bins(data_cfg["val_bin"], data_dir), block_size, random=False)
        monitor = "val_loss"

    datamodule = PretrainDataModule(
        train_ds,
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        train_group_weights=train_group_weights,
    )

    # --- model ---
    size = model_cfg.pop("size", "pico")
    overrides = {k: model_cfg[k] for k in MODEL_OVERRIDES if k in model_cfg}
    lm = build_lm(size, vocab_size=vocab_size, max_seq_len=max_seq_len, **overrides)
    # Recipe for rebuilding this exact architecture later (see GPT.model_config);
    # saved into the checkpoint so scripts/base_chat.py doesn't need it
    # repeated on the command line.
    model_config = dict(size=size, vocab_size=vocab_size, max_seq_len=max_seq_len, **overrides)

    # Config values are per-GPU (batch_size is per-device, lr/max_steps are
    # tuned for a single-device effective batch). Scale them by the device
    # count so a multi-GPU run matches the single-GPU training dynamics the
    # config was written for: lr scales up linearly with the larger effective
    # batch (linear scaling rule), and max_steps scales down so the run still
    # sees the same total number of tokens.
    num_devices = resolve_num_devices(args.devices)
    base_lr = optim_cfg.get("lr", 3e-4)
    base_muon_lr = optim_cfg.get("muon_lr", 0.02)
    base_max_steps = optim_cfg.get("max_steps", 10000)
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
    gpt = GPT(
        lm,
        pad_idx=pad_idx,
        lr=lr,
        weight_decay=optim_cfg.get("weight_decay", 0.1),
        # "muon" (default): Muon for hidden matrices + embedded AdamW for the
        # rest. "adamw": plain AdamW for everything.
        optimizer=optim_cfg.get("optimizer", "muon"),
        muon_lr=muon_lr,
        muon_momentum=optim_cfg.get("muon_momentum", 0.95),
        # GPT uses manual optimization (see GPT.__init__), so gradient clipping
        # and accumulation are owned by the module, not the Trainer.
        grad_clip=trainer_cfg.get("grad_clip", 1.0),
        accumulate=trainer_cfg.get("accumulate", 1),
        warmup_steps=optim_cfg.get("warmup_steps", 2000),
        max_steps=max_steps,
        # None -> auto (compile only where torch.compile is supported).
        compile=trainer_cfg.get("compile", None),
        # Used to render generated-text samples to TensorBoard during validation.
        tokenizer=tokenizer,
        model_config=model_config,
    )

    # --- resume: continue this exact stage (weights + optimizer + step) if it
    # already has a checkpoint in output_dir. Takes priority over init_from,
    # which only warm-starts weights for a *new* stage.
    resume_ckpt = Path(output_dir) / "last.ckpt"
    if resume_ckpt.exists():
        print(f"resuming stage from {resume_ckpt}", flush=True)
    else:
        resume_ckpt = None
        # --- continual learning: warm-start weights from a previous stage ---
        if init_from:
            ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if init_expand:
                stats = load_state_dict_expand(gpt, state)
                print(
                    f"warm-started weights from {init_from} (expand mode: "
                    f"{stats['matched']} matched, {stats['expanded']} expanded, "
                    f"{stats['skipped']} left at random init)",
                    flush=True,
                )
            else:
                gpt.load_state_dict(state)
                print(f"warm-started weights from {init_from}", flush=True)

    val_check_interval = trainer_cfg.get("val_check_interval", 500)
    ckpt_cb = ModelCheckpoint(
        dirpath=output_dir,
        filename="picochat-{step}",
        monitor=monitor,
        save_last=True,
        save_top_k=3 if monitor else 1,
        every_n_train_steps=val_check_interval if monitor is None else None,
    )

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        max_steps=max_steps,
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        # gradient_clip_val / accumulate_grad_batches are intentionally omitted:
        # GPT does manual optimization and Lightning forbids the Trainer from
        # managing them in that mode. They are passed to GPT above instead.
        val_check_interval=val_check_interval if val_ds is not None else 1.0,
        callbacks=[ckpt_cb],
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainer.fit(gpt, datamodule=datamodule, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
