"""Tokenize HF chat datasets (e.g. HuggingFaceTB/smoltalk) into padded SFT
tensors ready for picochat.data.sft.SFTDataset.

Every conversation is turned into a fixed-length (input_ids, labels) pair via
picochat.data.sft.encode_conversation (turn-by-turn tokenization, <pad>-based
loss masking -- see that module), and every surviving conversation is stacked
into a single {input_ids, labels, pad_id} tensor bundle saved as one .pt file.
SFT corpora are small enough to fit in memory, unlike base_setup.py's sharded
token-stream binaries for pretraining, so no shard directory is needed here.

Two ways to run:

  # one dataset, ad-hoc
  python scripts/sft_setup.py -p smoltalk -o data/sft/smoltalk.pt

  # many datasets from a recipe (configs/sft_setup/*.yml)
  python scripts/sft_setup.py --config configs/sft_setup/stage1.yml
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from picochat.data.sft import (
    PRESETS,
    ChatDatasetSpec,
    encode_conversation,
    iter_conversations,
    resolve_spec,
)
from picochat.tokenizer import load_tokenizer

# Used by the ad-hoc single-dataset mode; config mode reads the path from the
# recipe's `tokenizer:` field instead.
DEFAULT_TOKENIZER = "weights/tokenizer.json"
DEFAULT_MAX_LENGTH = 2048


def load_enc_and_pad_id(tokenizer_path: str):
    enc = load_tokenizer(tokenizer_path)
    return enc, enc.encode_single_token("<pad>")


def spec_from_entry(entry: dict) -> ChatDatasetSpec:
    """Resolve one `datasets:` entry into a ChatDatasetSpec.

    Either {preset: <name>} referencing picochat.data.sft, or an inline
    {path, name, split, messages_key}. An optional `split` overrides the preset's.
    """
    if "preset" in entry:
        name = entry["preset"]
        if name not in PRESETS:
            raise SystemExit(f"unknown preset '{name}'. choices: {', '.join(PRESETS)}")
        spec = PRESETS[name]
    elif "path" in entry:
        spec = ChatDatasetSpec(
            path=entry["path"],
            name=entry.get("name"),
            split=entry.get("split", "train"),
            messages_key=entry.get("messages_key", "messages"),
        )
    else:
        raise SystemExit(f"dataset entry needs 'preset' or 'path': {entry}")
    # Per-entry `split` override. Use replace() to copy the spec rather than
    # mutate it: PRESETS entries are shared, so mutating would leak the split
    # into every other entry using the same preset.
    if "split" in entry:
        spec = replace(spec, split=entry["split"])
    return spec


def process(
    spec: ChatDatasetSpec,
    output: Path,
    enc,
    pad_id: int,
    max_length: int,
    streaming: bool = True,
    limit: int | None = None,
) -> tuple[int, int]:
    """Tokenize every conversation of `spec` and save the surviving ones as a
    single (input_ids, labels) tensor pair at `output`. Returns (n_kept,
    n_dropped); a conversation is dropped when truncation to max_length left
    no assistant turn to train on (see encode_conversation).
    """
    all_input_ids: list[list[int]] = []
    all_labels: list[list[int]] = []
    n_dropped = 0
    bar = tqdm(desc=str(output))
    for messages in iter_conversations(spec, streaming=streaming, limit=limit):
        encoded = encode_conversation(messages, enc, max_length, pad_id)
        if encoded is None:
            n_dropped += 1
            continue
        input_ids, labels = encoded
        all_input_ids.append(input_ids)
        all_labels.append(labels)
        bar.update(1)
    bar.close()
    if not all_input_ids:
        raise SystemExit(f"no usable conversations from {spec.path} ({spec.split})")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "pad_id": pad_id,
        },
        output,
    )
    n_kept = len(all_input_ids)
    size_mb = output.stat().st_size / 1e6
    print(
        f"done: {n_kept:,} kept, {n_dropped:,} dropped -> {output} ({size_mb:.1f} MB)"
    )
    return n_kept, n_dropped


def run_config(cfg: dict, enc, pad_id: int) -> None:
    output_dir = Path(cfg.get("output_dir", ""))
    streaming = cfg.get("streaming", True)
    max_length = cfg.get("max_length", DEFAULT_MAX_LENGTH)
    entries = cfg["datasets"]
    for entry in entries:
        if "output" not in entry:
            raise SystemExit(f"dataset entry needs 'output': {entry}")

    for i, entry in enumerate(entries, 1):
        spec = spec_from_entry(entry)
        output = output_dir / entry["output"]
        limit = entry.get("limit")
        # Per-entry overrides of the file defaults.
        entry_streaming = entry.get("streaming", streaming)
        entry_max_length = entry.get("max_length", max_length)
        print(
            f"[{i}/{len(entries)}] {spec.path} ({spec.split}) -> {output}", flush=True
        )
        process(
            spec,
            output,
            enc,
            pad_id,
            entry_max_length,
            streaming=entry_streaming,
            limit=limit,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="preprocess recipe (YAML)"
    )
    # Single-dataset (ad-hoc) mode; ignored when --config is given.
    parser.add_argument("-o", "--output", type=str, default=None, help="output .pt file")
    parser.add_argument("-p", "--preset", type=str, default=None)
    parser.add_argument(
        "-d", "--dataset", type=str, default=None,
        help='inline spec: "path[:name[:split[:messages_key]]]"',
    )
    parser.add_argument(
        "-s", "--split", type=str, default=None, help="override the spec's split"
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--limit", type=int, default=None)
    # Default is streaming: chat corpora are pulled from the Hub on demand and
    # tokenized as they arrive, unlike base_setup.py's non-streaming default.
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.set_defaults(streaming=True)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if "tokenizer" not in cfg:
            raise SystemExit(
                f"{args.config} needs a 'tokenizer:' field (path to tokenizer.json)"
            )
        enc, pad_id = load_enc_and_pad_id(cfg["tokenizer"])
        run_config(cfg, enc, pad_id)
        return

    if not args.output:
        raise SystemExit("either --config, or --output with --preset/--dataset")
    enc, pad_id = load_enc_and_pad_id(DEFAULT_TOKENIZER)
    spec = resolve_spec(args.preset, args.dataset)
    if args.split is not None:
        spec = replace(spec, split=args.split)
    process(
        spec,
        Path(args.output),
        enc,
        pad_id,
        args.max_length,
        streaming=args.streaming,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
