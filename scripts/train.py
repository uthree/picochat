"""preprocess.py が生成した .bin で TransformerLM を事前学習する。

学習ループ自体は書かず、GPT(LightningModule) を L.Trainer で回す。
勾配クリップ・混合精度・勾配累積・チェックポイントは Trainer 側で設定する。
"""

import argparse

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from picochat.data.pretrain import PackedDataset
from picochat.model.gpt import GPT, TransformerLM
from picochat.tokenizer import load_tokenizer


def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train-bin", type=str, required=True)
    p.add_argument("--val-bin", type=str, default=None)
    p.add_argument("--tokenizer", type=str, default="weights/tokenizer.json")
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    # model
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-groups", type=int, default=None)
    p.add_argument("--n-attn-layers", type=int, default=None)
    # optim / trainer
    p.add_argument("--lr", type=float, default=3e-4)
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

    lm = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_groups=args.n_groups,
        n_attn_layers=args.n_attn_layers,
    )
    gpt = GPT(lm, pad_idx=0, lr=args.lr)

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
