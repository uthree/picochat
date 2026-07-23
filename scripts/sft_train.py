"""Fine-tune a pretrained checkpoint on SFT chat data (see scripts/sft_setup.py)
from a YAML recipe.

    python scripts/sft_train.py --config configs/sft_train/stage1.yml

Unlike base_train.py (continual pretraining, optional init_from), SFT always
fine-tunes an existing pretrained checkpoint: init_from is required, and the
architecture is rebuilt from that checkpoint's own `model_config`
hyperparameter (see scripts/chat.py for the same pattern). An optional
`model:` section in this config can override a couple of fields on top of
that (see MODEL_OVERRIDES) -- e.g. to extend context length.
"""

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import ConcatDataset

from picochat.config import (
    check_strategy,
    load_config,
    resolve_num_devices,
    scale_for_devices,
)
from picochat.data.dataloader import PretrainDataModule, SFTTensorDataset
from picochat.data.multimodal import (
    MultimodalDataModule,
    MultimodalSFTDataset,
)
from picochat.model.multimodal import MediaAdapters, build_encoders
from picochat.training import SFTModule, load_lm_from_checkpoint
from picochat.tokenizer import PAD_TOKEN, load_tokenizer


def resolve_paths(entries, data_dir: str) -> tuple[list[str], list[float]]:
    """Split `data.datasets` (list of {path, weight}) into parallel .pt paths
    and weights, joining `data_dir` with each path."""
    paths = [str(Path(data_dir) / e["path"]) for e in entries]
    weights = [e.get("weight", 1.0) for e in entries]
    return paths, weights


def make_dataset(paths: list[str], weights: list[float] | None = None):
    """Build a (Concat)SFTTensorDataset from one or more .pt paths written by
    scripts/sft_setup.py. Returns (dataset, group_weights): one weight per
    source, consumed by GroupWeightedIndexSampler so each source's total
    sampling mass equals its configured weight regardless of its example
    count. Mirrors scripts/base_train.py's make_dataset.
    """
    parts = [SFTTensorDataset(p) for p in paths]
    if len(parts) == 1:
        if weights is not None:
            raise ValueError("train_weights requires more than one dataset entry")
        return parts[0], None
    dataset = ConcatDataset(parts)
    if weights is None:
        return dataset, None
    if len(weights) != len(parts):
        raise ValueError(
            f"train_weights has {len(weights)} entries but {len(parts)} datasets"
        )
    return dataset, list(weights)


# Fields under `model:` that override the checkpoint's own model_config, e.g.
# to extend context length via continual learning: RoPE's sin/cos tables are
# non-persistent buffers rebuilt from this at construction time, not part of
# the checkpoint, so raising it doesn't affect any learned tensor's shape
# and the plain load_state_dict below still applies cleanly. (rope_base is
# fixed at 1e6, large enough for any context length this project targets.)
MODEL_OVERRIDES = ("max_seq_len",)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="SFT stage recipe (YAML)")
    p.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="pretrained checkpoint to fine-tune (overrides config's init_from)",
    )
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument(
        "--devices", type=str, default="auto", help="e.g. 'auto', '2', or '0,1'"
    )
    p.add_argument(
        "--num-nodes", type=int, default=1, help="number of nodes for multi-node DDP"
    )
    # DDP only -- sharded strategies (fsdp/deepspeed) are rejected, see
    # picochat.config.check_strategy.
    p.add_argument("--strategy", type=str, default="auto", help="e.g. 'auto' or 'ddp'")
    args = p.parse_args()
    check_strategy(args.strategy)

    cfg = load_config(args.config)

    # Same seed on every rank; the data samplers decorrelate ranks themselves
    # by drawing from seed + rank (see PretrainDataModule).
    L.seed_everything(cfg.get("seed", 42), workers=True)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    optim_cfg = cfg.get("optim", {})
    trainer_cfg = cfg.get("trainer", {})
    tokenizer_path = cfg.get("tokenizer", "weights/tokenizer.json")
    output_dir = cfg.get("output_dir", "weights")
    init_from = args.init_from or cfg.get("init_from")
    if not init_from:
        raise SystemExit("sft_train.py requires init_from (a pretrained checkpoint)")

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.n_vocab
    pad_idx = tokenizer.encode_single_token(PAD_TOKEN)

    # --- model: rebuilt from the pretrained checkpoint's own architecture,
    # optionally overriding max_seq_len to extend context ---
    model_overrides = {k: model_cfg[k] for k in MODEL_OVERRIDES if k in model_cfg}
    lm, model_config = load_lm_from_checkpoint(
        init_from, vocab_size, overrides=model_overrides
    )
    print(f"fine-tuning from {init_from} (model_config: {model_config})", flush=True)

    # --- multimodal (optional): a `multimodal:` section attaches media
    # encoders (Whisper audio / SigLIP2 vision, or from-scratch) and switches
    # the data pipeline to part-structured JSONL conversations (see
    # picochat.data.multimodal); without it this is the plain text-SFT stage.
    mm_cfg = cfg.get("multimodal")
    audio_encoder = vision_encoder = mm_config = None
    if mm_cfg:
        # d_model read off the built model (model_config is the build_lm
        # recipe -- a preset name plus overrides, no explicit d_model).
        audio_encoder, vision_encoder, mm_config = build_encoders(
            mm_cfg, lm.embed.embedding_dim
        )

    # --- data ---
    batch_size = trainer_cfg.get("batch_size", 2)
    num_workers = trainer_cfg.get("num_workers", 4)
    val_ds = None
    monitor = None
    if mm_cfg:
        media = MediaAdapters.from_encoders(audio_encoder, vision_encoder)
        max_length = mm_cfg.get("max_length", model_cfg.get("max_seq_len", 4096))
        train_ds = MultimodalSFTDataset(
            mm_cfg["data"], tokenizer, media, max_length, pad_idx
        )
        if mm_cfg.get("val_data"):
            val_ds = MultimodalSFTDataset(
                mm_cfg["val_data"], tokenizer, media, max_length, pad_idx
            )
            monitor = "val_loss"
        datamodule = MultimodalDataModule(
            train_ds,
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=cfg.get("seed", 42),
        )
    else:
        data_dir = data_cfg.get("data_dir", "data")
        train_paths, train_weights = resolve_paths(data_cfg["datasets"], data_dir)
        train_ds, train_group_weights = make_dataset(
            train_paths, weights=train_weights if len(train_paths) > 1 else None
        )
        if data_cfg.get("val_path"):
            val_ds, _ = make_dataset([str(Path(data_dir) / data_cfg["val_path"])])
            monitor = "val_loss"

        datamodule = PretrainDataModule(
            train_ds,
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            train_group_weights=train_group_weights,
            seed=cfg.get("seed", 42),
        )

    # Same per-device-config / linear-scaling convention as base_train.py.
    world_size = resolve_num_devices(args.devices, args.accelerator) * args.num_nodes
    lr, muon_lr, max_steps, warmup_steps = scale_for_devices(
        optim_cfg,
        world_size,
        lr_default=1e-5,
        muon_lr_default=0.005,
        max_steps_default=2000,
        warmup_steps_default=100,
    )

    sft = SFTModule(
        lm,
        pad_idx=pad_idx,
        lr=lr,
        weight_decay=optim_cfg.get("weight_decay", 0.1),
        optimizer=optim_cfg.get("optimizer", "muon"),
        muon_lr=muon_lr,
        muon_momentum=optim_cfg.get("muon_momentum", 0.95),
        # Independent from weight_decay: torch.optim.Muon's decay is decoupled
        # like AdamW's, but muon_lr runs an order of magnitude+ above lr, so
        # reusing weight_decay as-is would over-decay Muon's params (see
        # LMTrainerMixin._init_trainer).
        muon_weight_decay=optim_cfg.get("muon_weight_decay", 0.01),
        grad_clip=trainer_cfg.get("grad_clip", 1.0),
        accumulate=trainer_cfg.get("accumulate", 1),
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        compile=trainer_cfg.get("compile", None),
        # Opt-in memory saver: Liger fused cross-entropy (see picochat.training.kernels
        # and the matching comment in base_train.py).
        fused_loss=trainer_cfg.get("fused_loss", False),
        tokenizer=tokenizer,
        model_config=model_config,
        # Auxiliary multi-token-prediction loss weight (n_mtp > 0 models only).
        mtp_weight=optim_cfg.get("mtp_weight", 0.3),
        audio_encoder=audio_encoder,
        vision_encoder=vision_encoder,
        # Stage-1 default: pretrained towers frozen, projectors (+ LM) train;
        # flip for a later full finetune.
        train_towers=bool(mm_cfg.get("train_towers", False)) if mm_cfg else False,
        mm_config=mm_config,
    )

    # --- resume: continue this exact SFT stage (weights + optimizer + step)
    # if it already has a checkpoint in output_dir.
    resume_ckpt = Path(output_dir) / "last.ckpt"
    if resume_ckpt.exists():
        print(f"resuming SFT stage from {resume_ckpt}", flush=True)
    else:
        resume_ckpt = None

    val_check_interval = trainer_cfg.get("val_check_interval", 200)
    ckpt_cb = ModelCheckpoint(
        dirpath=output_dir,
        filename="picochat-sft-{step}",
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
        # gradient_clip_val / accumulate_grad_batches are intentionally omitted:
        # SFTModule does manual optimization and Lightning forbids the Trainer
        # from managing them in that mode. They are passed to SFTModule above.
        val_check_interval=val_check_interval if val_ds is not None else 1.0,
        # The chunked train samplers are rank-aware themselves and the val
        # loader builds its own DistributedSampler (see PretrainDataModule).
        use_distributed_sampler=False,
        callbacks=[ckpt_cb],
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainer.fit(sft, datamodule=datamodule, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
