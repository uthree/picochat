"""Multimodal SFT data: part-structured conversations with audio/image files.

On-disk format: a JSONL file, one conversation per line --

    {"messages": [
        {"role": "user", "content": [
            {"type": "audio", "path": "clips/q1.wav"},
            {"type": "text", "text": "What is said here?"}]},
        {"role": "assistant", "content": "..."}]}

Media `path`s are resolved relative to `media_root` (default: the JSONL's own
directory). Audio must decode to mono 16 kHz (other rates are resampled);
images are anything PIL opens.

Unlike text SFT (scripts/sft_setup.py pre-tokenizes into packed tensors),
multimodal conversations are NOT packed: each batch row is one padded
conversation, because the media features must stay aligned with their
placeholder spans and are produced at load time by the encoders' processors.
The dataset tokenizes every conversation once at construction (media must be
decoded to know each clip's placeholder count) and re-derives the features in
__getitem__ -- the mel/pixel pipeline is deterministic, so the counts always
match the cached token ids.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import lightning as L
import torch
from torch import Tensor

from picochat.model.multimodal import MediaAdapters, encode_mm_conversation
from picochat.tokenizer import Tokenizer


def _load_audio(path: Path, sample_rate: int) -> Tensor:
    """Decode an audio file to a mono waveform at `sample_rate` (1-D float).

    soundfile (bundled libsndfile) does the decoding -- torchaudio.load now
    delegates to the optional torchcodec/ffmpeg stack, which this pipeline
    deliberately avoids depending on; torchaudio's pure-torch resampler is
    still used for rate conversion."""
    import soundfile
    import torchaudio

    data, sr = soundfile.read(str(path), dtype="float32")
    wav = torch.from_numpy(data)
    if wav.ndim == 2:
        wav = wav.mean(1)  # downmix to mono
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def _load_image(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _resolve_media(messages: list[dict], media_root: Path, sample_rate: int):
    """Yield a deep-ish copy of `messages` with media paths decoded into
    tensors/images (the part dicts render_parts consumes)."""
    out = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append(msg)
            continue
        parts = []
        for part in content:
            if part["type"] == "audio" and "path" in part:
                parts.append(
                    {
                        "type": "audio",
                        "audio": _load_audio(media_root / part["path"], sample_rate),
                    }
                )
            elif part["type"] == "image" and "path" in part:
                parts.append(
                    {"type": "image", "image": _load_image(media_root / part["path"])}
                )
            else:
                parts.append(part)
        out.append({**msg, "content": parts})
    return out


def load_mm_conversations(jsonl_path: os.PathLike) -> list[list[dict]]:
    """Read the JSONL conversation records (media paths left unresolved)."""
    conversations = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line)["messages"])
    return conversations


class MultimodalSFTDataset(torch.utils.data.Dataset):
    """Part-structured SFT conversations -> (input_ids, labels, mels, images).

    Tokenizes everything once at construction (dropping conversations
    encode_mm_conversation rejects, e.g. nothing trainable within max_length)
    and decodes media again per __getitem__; see the module docstring for why.
    """

    def __init__(
        self,
        jsonl_path: os.PathLike,
        tokenizer: Tokenizer,
        media: MediaAdapters,
        max_length: int,
        pad_id: int,
        media_root: os.PathLike | None = None,
    ):
        self.tokenizer = tokenizer
        self.media = media
        self.max_length = max_length
        self.pad_id = pad_id
        self.media_root = Path(
            media_root if media_root is not None else Path(jsonl_path).parent
        )
        self.sample_rate = (
            media.audio_processor.cfg.sample_rate if media.audio_processor else 16000
        )
        self.items: list[list[dict]] = []
        dropped = 0
        for messages in load_mm_conversations(jsonl_path):
            resolved = _resolve_media(messages, self.media_root, self.sample_rate)
            if encode_mm_conversation(resolved, tokenizer, max_length, pad_id, media):
                self.items.append(messages)
            else:
                dropped += 1
        if dropped:
            print(f"dropped {dropped} conversations with nothing to train on")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        resolved = _resolve_media(self.items[idx], self.media_root, self.sample_rate)
        encoded = encode_mm_conversation(
            resolved, self.tokenizer, self.max_length, self.pad_id, self.media
        )
        input_ids, labels, mels, images = encoded
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "mels": mels,
            "images": images,
        }


def mm_collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad conversations to the batch max and gather the media lists in
    row-major order (the order scatter_embeds expects). `doc_ids` gives every
    pad position its own segment id -- the same convention pretraining uses
    for pad tails -- so neither mixer attends from the padding back into the
    conversation."""
    length = max(item["input_ids"].shape[0] for item in batch)

    def pad(t: Tensor) -> Tensor:
        return torch.nn.functional.pad(t, (0, length - t.shape[0]), value=pad_id)

    input_ids = torch.stack([pad(item["input_ids"]) for item in batch])
    labels = torch.stack([pad(item["labels"]) for item in batch])
    mels = [mel for item in batch for mel in item["mels"]]
    images = [img for item in batch for img in item["images"]]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "doc_ids": (input_ids == pad_id).cumsum(-1),
        "mels": mels,
        "images": images,
    }


class MultimodalDataModule(L.LightningDataModule):
    """Train/val loaders over MultimodalSFTDataset with the mm_collate above.

    Under DDP each rank gets a DistributedSampler shard (the packed text
    pipeline's chunked samplers don't apply here -- multimodal batches are
    unpacked, one conversation per row)."""

    def __init__(
        self,
        train_ds: MultimodalSFTDataset,
        val_ds: MultimodalSFTDataset | None = None,
        batch_size: int = 2,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def _loader(self, ds: MultimodalSFTDataset, shuffle: bool):
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
            collate_fn=lambda batch: mm_collate(batch, ds.pad_id),
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        if self.val_ds is None:
            return []
        return self._loader(self.val_ds, shuffle=False)
