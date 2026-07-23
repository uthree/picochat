"""Direct Preference Optimization (arXiv:2305.18290): preference tuning for
general chat quality, the stage GRPO's verifiable rewards cannot cover.

Sits between SFT and GRPO in the post-training recipe (SFT -> DPO -> GRPO):
DPO needs only (prompt, chosen, rejected) text pairs and a frozen reference
model -- no reward model, no rollouts -- so it is the cheap way to push broad
"which answer is better" preferences. Pairs come from a JSONL file (see
PreferenceDataset; scripts/dpo_setup.py can bootstrap one by sampling the
policy twice per prompt and letting the LLM judge pick the winner).

The loss is the standard sigmoid DPO objective on sequence log-probs summed
over response tokens:

    L = -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))

The reference is the init_from checkpoint itself (the SFT policy), stored
outside the state_dict exactly like GRPO stores its reference -- it is
reconstructable from init_from, so checkpoints don't double in size.
"""

from __future__ import annotations

import json
import os

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor

from picochat.model.transformer import TransformerLM
from picochat.tokenizer import Tokenizer, encode_conversation
from picochat.training.modules import LMTrainerMixin


def sequence_logprobs(model: TransformerLM, ids: Tensor, labels: Tensor) -> Tensor:
    """Sum of response-token log-probs per sequence: position t scores
    ids[:, t+1] and counts iff labels[:, t+1] is a real target (not the
    pad/ignore id encoded as -100 here). Returns (B,). Mirrors
    picochat.rl.grpo.token_logprobs, reduced over the labeled positions."""
    logits = model(ids)  # (B, L, V)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = ids[:, 1:]
    mask = labels[:, 1:] != -100
    picked = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(-1)


def dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    ref_chosen: Tensor,
    ref_rejected: Tensor,
    beta: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """Sigmoid DPO over per-sequence log-probs (all (B,)). Returns the mean
    loss and logging metrics: `margin` (mean implicit-reward gap, the thing
    DPO grows) and `accuracy` (how often the chosen response is already
    preferred)."""
    chosen_reward = policy_chosen - ref_chosen
    rejected_reward = policy_rejected - ref_rejected
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(beta * margin).mean()
    return loss, {
        "margin": float(margin.mean()),
        "accuracy": float((margin > 0).float().mean()),
    }


class PreferenceDataset(torch.utils.data.Dataset):
    """(prompt, chosen, rejected) pairs from a JSONL file:

        {"prompt": [{"role": "user", "content": "..."}],
         "chosen": "better reply", "rejected": "worse reply"}

    `prompt` is a ChatML message list (a plain string is wrapped as one user
    turn). Each side is tokenized with tokenizer.encode_conversation --
    prompt turns loss-masked, the reply body trainable -- so the DPO
    log-probs cover exactly the tokens SFT would have trained. Pairs whose
    replies don't survive max_length are dropped at construction."""

    def __init__(
        self,
        jsonl_path: os.PathLike,
        tokenizer: Tokenizer,
        max_length: int,
        pad_id: int,
    ):
        self.items: list[tuple[list[int], list[int], list[int], list[int]]] = []
        dropped = 0
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        for rec in records:
            prompt = rec["prompt"]
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            sides = []
            for reply in (rec["chosen"], rec["rejected"]):
                messages = prompt + [{"role": "assistant", "content": reply}]
                sides.append(
                    encode_conversation(messages, tokenizer, max_length, pad_id)
                )
            if any(side is None for side in sides):
                dropped += 1
                continue
            (c_ids, c_labels), (r_ids, r_labels) = sides
            self.items.append((c_ids, c_labels, r_ids, r_labels))
        if dropped:
            print(f"dropped {dropped} pairs that don't fit max_length")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        c_ids, c_labels, r_ids, r_labels = self.items[idx]
        return {
            "chosen_ids": torch.tensor(c_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(c_labels, dtype=torch.long),
            "rejected_ids": torch.tensor(r_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(r_labels, dtype=torch.long),
        }


def dpo_collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad each side to its batch max. Labels are re-encoded to -100 at
    every masked/padding position (encode_conversation emits pad_id there;
    sequence_logprobs must not treat a *real* pad-id token in the input as a
    target either, and -100 disambiguates)."""

    def pad_stack(rows: list[Tensor], value: int) -> Tensor:
        length = max(r.shape[0] for r in rows)
        return torch.stack(
            [F.pad(r, (0, length - r.shape[0]), value=value) for r in rows]
        )

    out = {}
    for side in ("chosen", "rejected"):
        ids = pad_stack([item[f"{side}_ids"] for item in batch], pad_id)
        labels = pad_stack([item[f"{side}_labels"] for item in batch], pad_id)
        labels = labels.masked_fill(labels == pad_id, -100)
        out[f"{side}_ids"] = ids
        out[f"{side}_labels"] = labels
    return out


class DPOModule(LMTrainerMixin, L.LightningModule):
    """Preference tuning of `transformer_lm` against a frozen `reference_lm`
    (the same init_from weights). Reuses the SFT optimizer scaffolding
    (Muon/AdamW split, warmup+cosine, manual optimization) via
    LMTrainerMixin; only the loss differs."""

    def __init__(
        self,
        transformer_lm: TransformerLM,
        reference_lm: TransformerLM,
        pad_idx: int,
        beta: float = 0.1,
        lr: float = 5e-7,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        optimizer: str = "adamw",
        muon_lr: float = 0.002,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.01,
        warmup_steps: int = 50,
        max_steps: int | None = None,
        min_lr_ratio: float = 0.1,
        grad_clip: float | None = 1.0,
        accumulate: int = 1,
        tokenizer=None,
        model_config: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters("model_config")
        self.model = transformer_lm
        self.pad_idx = pad_idx
        self.beta = beta
        # Frozen reference, kept out of state_dict/optimizer exactly like
        # GRPOModule does it (reconstructable from init_from).
        reference_lm.requires_grad_(False).eval()
        object.__setattr__(self, "reference", reference_lm)
        self._init_trainer(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            optimizer=optimizer,
            muon_lr=muon_lr,
            muon_momentum=muon_momentum,
            muon_weight_decay=muon_weight_decay,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
            grad_clip=grad_clip,
            accumulate=accumulate,
            tokenizer=tokenizer,
            fused_loss=False,
        )
        self.automatic_optimization = False
        # No compiled forward here: the DPO loss runs four plain forwards per
        # step (policy/reference x chosen/rejected) through sequence_logprobs,
        # not the mixin's _forward path.

    def on_fit_start(self) -> None:
        self.reference.to(self.device)

    def _loss(self, batch: dict) -> tuple[Tensor, dict[str, float]]:
        pc = sequence_logprobs(self.model, batch["chosen_ids"], batch["chosen_labels"])
        pr = sequence_logprobs(
            self.model, batch["rejected_ids"], batch["rejected_labels"]
        )
        with torch.no_grad():
            rc = sequence_logprobs(
                self.reference, batch["chosen_ids"], batch["chosen_labels"]
            )
            rr = sequence_logprobs(
                self.reference, batch["rejected_ids"], batch["rejected_labels"]
            )
        return dpo_loss(pc, pr, rc, rr, beta=self.beta)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, metrics = self._loss(batch)
        self._backward_and_step(loss, batch_idx)
        for name, value in metrics.items():
            self.log(f"dpo/{name}", value)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, metrics = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        for name, value in metrics.items():
            self.log(f"val_dpo/{name}", value, sync_dist=True)
        return loss


class PreferenceDataModule(L.LightningDataModule):
    """Train/val loaders over PreferenceDataset with dpo_collate; a
    DistributedSampler shard per rank under DDP (pairs are unpacked rows,
    like the multimodal SFT loader)."""

    def __init__(
        self,
        train_ds: PreferenceDataset,
        val_ds: PreferenceDataset | None,
        pad_id: int,
        batch_size: int = 4,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.pad_id = pad_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def _loader(self, ds: PreferenceDataset, shuffle: bool):
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                ds, shuffle=shuffle, seed=self.seed
            )
            shuffle = False
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=lambda batch: dpo_collate(batch, self.pad_id),
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        if self.val_ds is None:
            return []
        return self._loader(self.val_ds, shuffle=False)
