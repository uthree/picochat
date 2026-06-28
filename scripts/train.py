"""Pretrain a TransformerLM on the .bin produced by preprocess.py.

We do not write a training loop ourselves; we run GPT(LightningModule) with an
L.Trainer. Gradient clipping, mixed precision, gradient accumulation, and
checkpointing are all configured on the Trainer side.
"""

import argparse

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from picochat.data.pretrain import PackedDataset
from picochat.model.gpt import GPT, MODEL_PRESETS, build_lm
from picochat.tokenizer import load_tokenizer


def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train-bin", type=str, required=True)
    p.add_argument("--val-bin", type=str, default=None)
    p.add_argument("--tokenizer", type=str, default="weights/tokenizer.json")
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    # model: pick a scale-ladder preset with --size, override individual fields if needed
    p.add_argument(
        "--size", type=str, default="pico", help=f"scale {list(MODEL_PRESETS)}"
    )
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--n-groups", type=int, default=None)
    # optim / trainer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--precision", type=str, default="bf16-mixed")
    p.add_argument("--val-check-interval", type=int, default=500)
    p.add_argument("--ckpt-dir", type=str, default="weights")
    p.add_argument("--accelerator", type=str, default="auto")
    args = p.parse_args()

    vocab_size = load_tokenizer(args.tokenizer).n_vocab

    train_ds = PackedDataset(args.train_bin, block_size=args.block_size, random=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=True,
    )

    val_dl = None
    monitor = None
    if args.val_bin is not None:
        val_ds = PackedDataset(args.val_bin, block_size=args.block_size, random=False)
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
        )
        monitor = "val_loss"

    # The model processes block_size+1 tokens per position, so it must fit in max_seq_len.
    assert args.block_size < args.max_seq_len, "block_size+1 <= max_seq_len required"
    overrides = {
        k: v
        for k, v in {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "n_groups": args.n_groups,
        }.items()
        if v is not None
    }
    lm = build_lm(
        args.size,
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        **overrides,
    )
    gpt = GPT(
        lm,
        pad_idx=0,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    ckpt = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="picochat-{step}",
        monitor=monitor,
        save_last=True,
        save_top_k=3 if monitor else 1,
        every_n_train_steps=args.val_check_interval if monitor is None else None,
    )

    trainer = L.Trainer(
        accelerator=args.accelerator,
        max_steps=args.max_steps,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate,
        val_check_interval=args.val_check_interval if val_dl is not None else 1.0,
        callbacks=[ckpt],
    )
    trainer.fit(gpt, train_dl, val_dl)


if __name__ == "__main__":
    main()
