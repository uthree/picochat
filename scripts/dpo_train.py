"""Preference-tune an SFT checkpoint with DPO from a YAML recipe.

    python scripts/dpo_train.py --config configs/dpo/stage.yml

The recipe mirrors sft_train.py: `init_from` names the SFT checkpoint (it
becomes both the trainable policy and the frozen reference), `data` points at
a preference-pair JSONL (see picochat.training.dpo.PreferenceDataset for the
format; scripts/dpo_setup.py bootstraps one with the LLM judge). The
recommended pipeline position is SFT -> DPO -> GRPO.
"""

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from picochat.config import (
    check_strategy,
    load_config,
    resolve_num_devices,
    scale_for_devices,
)
from picochat.tokenizer import PAD_TOKEN, load_tokenizer
from picochat.training import load_lm_from_checkpoint
from picochat.training.callbacks import benchmark_callback_from_config
from picochat.training.dpo import (
    DPOModule,
    PreferenceDataModule,
    PreferenceDataset,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="DPO recipe (YAML)")
    p.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="SFT checkpoint to preference-tune (overrides config's init_from)",
    )
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto")
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--strategy", type=str, default="auto")
    args = p.parse_args()
    check_strategy(args.strategy)

    cfg = load_config(args.config)
    L.seed_everything(cfg.get("seed", 42), workers=True)

    data_cfg = cfg.get("data", {})
    optim_cfg = cfg.get("optim", {})
    trainer_cfg = cfg.get("trainer", {})
    tokenizer_path = cfg.get("tokenizer", "weights/tokenizer.json")
    output_dir = cfg.get("output_dir", "weights/dpo")
    init_from = args.init_from or cfg.get("init_from")
    if not init_from:
        raise SystemExit("dpo_train.py requires init_from (an SFT checkpoint)")

    tokenizer = load_tokenizer(tokenizer_path)
    pad_idx = tokenizer.encode_single_token(PAD_TOKEN)

    # Policy and frozen reference start as the same weights; load the ckpt
    # once and build two models from it.
    ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
    lm, model_config = load_lm_from_checkpoint(init_from, tokenizer.n_vocab, ckpt=ckpt)
    reference, _ = load_lm_from_checkpoint(init_from, tokenizer.n_vocab, ckpt=ckpt)
    print(f"DPO from {init_from} (model_config: {model_config})", flush=True)

    max_length = data_cfg.get("max_length", 4096)
    train_ds = PreferenceDataset(data_cfg["train"], tokenizer, max_length, pad_idx)
    val_ds = None
    monitor = None
    if data_cfg.get("val"):
        val_ds = PreferenceDataset(data_cfg["val"], tokenizer, max_length, pad_idx)
        monitor = "val_loss"
    print(f"{len(train_ds)} training pairs", flush=True)

    datamodule = PreferenceDataModule(
        train_ds,
        val_ds,
        pad_id=pad_idx,
        batch_size=trainer_cfg.get("batch_size", 4),
        num_workers=trainer_cfg.get("num_workers", 2),
        seed=cfg.get("seed", 42),
    )

    world_size = resolve_num_devices(args.devices, args.accelerator) * args.num_nodes
    lr, muon_lr, max_steps, warmup_steps = scale_for_devices(
        optim_cfg,
        world_size,
        lr_default=5e-7,
        muon_lr_default=0.002,
        max_steps_default=1000,
        warmup_steps_default=50,
    )

    dpo = DPOModule(
        lm,
        reference,
        pad_idx=pad_idx,
        beta=optim_cfg.get("beta", 0.1),
        lr=lr,
        weight_decay=optim_cfg.get("weight_decay", 0.0),
        # AdamW by default: DPO nudges an already-tuned policy with tiny LRs,
        # where Muon's aggressive hidden-matrix updates are unnecessary risk.
        optimizer=optim_cfg.get("optimizer", "adamw"),
        muon_lr=muon_lr,
        muon_momentum=optim_cfg.get("muon_momentum", 0.95),
        muon_weight_decay=optim_cfg.get("muon_weight_decay", 0.01),
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        grad_clip=trainer_cfg.get("grad_clip", 1.0),
        accumulate=trainer_cfg.get("accumulate", 1),
        tokenizer=tokenizer,
        model_config=model_config,
    )

    bench_cb = benchmark_callback_from_config(trainer_cfg, tokenizer, chat=True)

    resume_ckpt = Path(output_dir) / "last.ckpt"
    resume_ckpt = resume_ckpt if resume_ckpt.exists() else None
    if resume_ckpt:
        print(f"resuming DPO stage from {resume_ckpt}", flush=True)

    val_check_interval = trainer_cfg.get("val_check_interval", 100)
    ckpt_cb = ModelCheckpoint(
        dirpath=output_dir,
        filename="picochat-dpo-{step}",
        monitor=monitor,
        save_last=True,
        save_top_k=3 if monitor else 1,
        every_n_train_steps=val_check_interval if monitor is None else None,
    )

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        max_steps=max_steps,
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        val_check_interval=val_check_interval if val_ds is not None else 1.0,
        use_distributed_sampler=False,
        callbacks=[cb for cb in (ckpt_cb, bench_cb) if cb is not None],
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.fit(dpo, datamodule=datamodule, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
