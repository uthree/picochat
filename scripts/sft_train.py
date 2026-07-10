"""Fine-tune a pretrained checkpoint on SFT chat data (see scripts/sft_setup.py)
from a YAML recipe.

    python scripts/sft_train.py --config configs/sft_train/stage1.yml

Unlike base_train.py (continual pretraining, optional init_from), SFT always
fine-tunes an existing pretrained checkpoint: init_from is required, and the
architecture is rebuilt from that checkpoint's own `model_config`
hyperparameter (see scripts/base_chat.py for the same pattern). An optional
`model:` section in this config can override a couple of fields on top of
that (see MODEL_OVERRIDES) -- e.g. to extend context length.
"""

import argparse
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import ConcatDataset

from picochat.data.pretrain import PretrainDataModule
from picochat.data.sft import SFTTensorDataset
from picochat.model.gpt import build_lm
from picochat.model.sft import SFTModule
from picochat.tokenizer import PAD_TOKEN, load_tokenizer


def resolve_num_devices(devices: str) -> int:
    """Single-node device count implied by --devices, for scaling lr/max_steps.

    Mirrors scripts/base_train.py's helper of the same name.
    """
    if devices == "auto":
        return torch.cuda.device_count() if torch.cuda.is_available() else 1
    if "," in devices:
        return len(devices.split(","))
    return int(devices)


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
# non-persistent buffers rebuilt from these at construction time, not part of
# the checkpoint, so raising them doesn't affect any learned tensor's shape
# and the plain load_state_dict below still applies cleanly.
MODEL_OVERRIDES = ("rope_base", "max_seq_len")


def load_pretrained(checkpoint: str, vocab_size: int, overrides: dict | None = None):
    """Rebuild the TransformerLM architecture from a checkpoint's own
    `model_config` hyperparameter (see scripts/base_chat.py), apply `overrides`
    on top, and load the checkpoint's weights. Returns (TransformerLM, model_config).
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"{checkpoint} doesn't look like a Lightning checkpoint")
    model_config = (ckpt.get("hyper_parameters") or {}).get("model_config")
    if model_config is None:
        raise ValueError(
            f"{checkpoint} has no 'model_config' hyperparameter -- it predates "
            "GPT.__init__ saving it, so its architecture can't be rebuilt."
        )
    model_config = {**model_config, **(overrides or {})}
    lm = build_lm(**{**model_config, "vocab_size": vocab_size})
    # GPT's state_dict keys are "model.*" (the wrapped TransformerLM) plus the
    # trainer scaffolding around it; strip the prefix to load into a bare lm.
    prefix = "model."
    state = {
        k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)
    }
    lm.load_state_dict(state)
    return lm, model_config


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
    p.add_argument("--devices", type=str, default="auto", help="e.g. 'auto', '2', or '0,1'")
    p.add_argument("--strategy", type=str, default="auto", help="e.g. 'auto', 'ddp', 'fsdp'")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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

    # --- data ---
    batch_size = trainer_cfg.get("batch_size", 2)
    num_workers = trainer_cfg.get("num_workers", 4)
    data_dir = data_cfg.get("data_dir", "data")
    train_paths, train_weights = resolve_paths(data_cfg["datasets"], data_dir)
    train_ds, train_group_weights = make_dataset(
        train_paths, weights=train_weights if len(train_paths) > 1 else None
    )
    val_ds = None
    monitor = None
    if data_cfg.get("val_path"):
        val_ds, _ = make_dataset([str(Path(data_dir) / data_cfg["val_path"])])
        monitor = "val_loss"

    datamodule = PretrainDataModule(
        train_ds,
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        train_group_weights=train_group_weights,
    )

    # --- model: rebuilt from the pretrained checkpoint's own architecture,
    # optionally overriding e.g. max_seq_len/rope_base to extend context ---
    model_overrides = {k: model_cfg[k] for k in MODEL_OVERRIDES if k in model_cfg}
    lm, model_config = load_pretrained(init_from, vocab_size, overrides=model_overrides)
    print(f"fine-tuning from {init_from} (model_config: {model_config})", flush=True)

    # Same per-device-config / linear-scaling convention as base_train.py.
    num_devices = resolve_num_devices(args.devices)
    base_lr = optim_cfg.get("lr", 1e-5)
    base_muon_lr = optim_cfg.get("muon_lr", 0.005)
    base_max_steps = optim_cfg.get("max_steps", 2000)
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

    sft = SFTModule(
        lm,
        pad_idx=pad_idx,
        lr=lr,
        weight_decay=optim_cfg.get("weight_decay", 0.1),
        optimizer=optim_cfg.get("optimizer", "muon"),
        muon_lr=muon_lr,
        muon_momentum=optim_cfg.get("muon_momentum", 0.95),
        grad_clip=trainer_cfg.get("grad_clip", 1.0),
        accumulate=trainer_cfg.get("accumulate", 1),
        warmup_steps=optim_cfg.get("warmup_steps", 100),
        max_steps=max_steps,
        compile=trainer_cfg.get("compile", None),
        tokenizer=tokenizer,
        model_config=model_config,
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
        strategy=args.strategy,
        max_steps=max_steps,
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        # gradient_clip_val / accumulate_grad_batches are intentionally omitted:
        # SFTModule does manual optimization and Lightning forbids the Trainer
        # from managing them in that mode. They are passed to SFTModule above.
        val_check_interval=val_check_interval if val_ds is not None else 1.0,
        callbacks=[ckpt_cb],
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainer.fit(sft, datamodule=datamodule, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
