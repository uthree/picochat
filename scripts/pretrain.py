"""Train one stage from a YAML recipe.

A stage = one training run defined entirely by a config file (model size, data
bins, optimizer, trainer). We do NOT orchestrate multiple stages here; to run a
curriculum, train one stage, then point the next stage's config at the produced
checkpoint via `init_from` to warm-start (continual learning) with a fresh
optimizer and LR schedule.

    python scripts/pretrain.py --config configs/pretrain/stage1_basic.yml

The model architecture must stay the same across stages that chain via
`init_from` (only the data / schedule change).
"""

import argparse
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import ConcatDataset, DataLoader

from picochat.data.pretrain import PackedDataset
from picochat.model.gpt import GPT, build_lm
from picochat.tokenizer import load_tokenizer

# Fields under `model:` that override the scale-ladder preset.
MODEL_OVERRIDES = (
    "d_model",
    "n_heads",
    "n_kv_heads",
    "n_layers",
    "tie_embeddings",
    "grad_checkpoint",
)


def make_dataset(bins, block_size: int, random: bool):
    """Build a (Concat)PackedDataset from a single path or a list of paths."""
    if isinstance(bins, str):
        bins = [bins]
    parts = [PackedDataset(b, block_size=block_size, random=random) for b in bins]
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="stage recipe (YAML)")
    p.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="checkpoint to warm-start from (overrides config's init_from)",
    )
    p.add_argument("--accelerator", type=str, default="auto")
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

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.n_vocab

    # --- data ---
    block_size = data_cfg.get("block_size", 1024)
    max_seq_len = model_cfg.pop("max_seq_len", 4096)
    assert block_size < max_seq_len, "block_size+1 <= max_seq_len required"
    batch_size = trainer_cfg.get("batch_size", 16)
    num_workers = trainer_cfg.get("num_workers", 4)

    train_ds = make_dataset(data_cfg["train_bin"], block_size, random=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True,
    )

    val_dl = None
    monitor = None
    if data_cfg.get("val_bin"):
        val_ds = make_dataset(data_cfg["val_bin"], block_size, random=False)
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )
        monitor = "val_loss"

    # --- model ---
    size = model_cfg.pop("size", "pico")
    overrides = {k: model_cfg[k] for k in MODEL_OVERRIDES if k in model_cfg}
    lm = build_lm(size, vocab_size=vocab_size, max_seq_len=max_seq_len, **overrides)

    max_steps = optim_cfg.get("max_steps", 10000)
    gpt = GPT(
        lm,
        pad_idx=0,
        lr=optim_cfg.get("lr", 3e-4),
        weight_decay=optim_cfg.get("weight_decay", 0.1),
        warmup_steps=optim_cfg.get("warmup_steps", 2000),
        max_steps=max_steps,
        # None -> auto (compile only where torch.compile is supported).
        compile=trainer_cfg.get("compile", None),
        # Used to render generated-text samples to TensorBoard during validation.
        tokenizer=tokenizer,
    )

    # --- continual learning: warm-start weights from a previous stage ---
    if init_from:
        ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
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
        max_steps=max_steps,
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=trainer_cfg.get("grad_clip", 1.0),
        accumulate_grad_batches=trainer_cfg.get("accumulate", 1),
        val_check_interval=val_check_interval if val_dl is not None else 1.0,
        callbacks=[ckpt_cb],
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.fit(gpt, train_dl, val_dl)


if __name__ == "__main__":
    main()
