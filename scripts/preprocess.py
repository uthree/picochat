"""Tokenize HF datasets and convert them into packed, flat token binaries.

Each document is encoded, an <eos> is appended, and everything is concatenated
into a single continuous token stream written to a .bin file. No padding is
added; the training side (PackedDataset) slices a block_size+1 window at read
time. Tokens are stored as uint32 (DTYPE), which fits vocab up to ~4.29B.

Two ways to run:

  # one dataset, ad-hoc
  python scripts/preprocess.py -p tinystories -o data/tinystories.bin

  # many datasets from a recipe (configs/preprocess/*.yml)
  python scripts/preprocess.py --config configs/preprocess/stage1_basic.yml
"""

import argparse
import os
import time
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml
from tqdm import tqdm

from datasets import get_dataset_split_names

from picochat.data.pretrain import DTYPE  # uint32; shared with the reader
from picochat.data.sources import PRESETS, DatasetSpec, iter_texts, resolve_spec
from picochat.tokenizer import load_tokenizer

EOS_TOKEN = "</s>"
# Encoding throughput is dominated by tiktoken. Encoding one doc at a time has
# heavy per-call overhead, so we feed it large batches: encode_ordinary_batch
# tokenizes them in parallel across Rust threads (GIL released).
BATCH_SIZE = 1024


def _batched(it: Iterator[str], n: int) -> Iterator[list[str]]:
    while batch := list(islice(it, n)):
        yield batch


def split_unavailable(spec: DatasetSpec) -> bool:
    """True only if we can confirm spec.split is absent for this dataset.

    Lets the caller skip an entry whose split doesn't exist (e.g. a validation
    entry for a train-only dataset like wikipedia) instead of crashing. If the
    splits can't be listed (gated/offline dataset), returns False so processing
    is still attempted.
    """
    try:
        available = get_dataset_split_names(spec.path, spec.name)
    except Exception:
        return False
    base = spec.split.split("[")[0]  # ignore slicing like "train[:1%]"
    return base not in available


def load_enc(tokenizer_path: str):
    """Load the tokenizer and return (encoding, eos_id), checking vocab fits DTYPE."""
    enc = load_tokenizer(tokenizer_path)
    eos_id = enc._special_tokens[EOS_TOKEN]
    assert enc.n_vocab <= np.iinfo(DTYPE).max + 1, (
        f"vocab {enc.n_vocab} does not fit in {DTYPE}"
    )
    return enc, eos_id


def spec_from_entry(entry: dict) -> DatasetSpec:
    """Resolve one `datasets:` entry into a DatasetSpec.

    Either {preset: <name>} referencing picochat.data.sources, or an inline
    {path, name, split, text_key}. An optional `split` overrides the preset's.
    """
    if "preset" in entry:
        name = entry["preset"]
        if name not in PRESETS:
            raise SystemExit(f"unknown preset '{name}'. choices: {', '.join(PRESETS)}")
        spec = PRESETS[name]
    elif "path" in entry:
        spec = DatasetSpec(
            path=entry["path"],
            name=entry.get("name"),
            split=entry.get("split", "train"),
            text_key=entry.get("text_key", "text"),
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
    spec: DatasetSpec,
    output: Path,
    enc,
    eos_id: int,
    streaming: bool = False,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
    num_threads: int | None = None,
) -> tuple[int, int]:
    """Encode every document of `spec` into `output`. Returns (n_docs, n_tokens).

    Documents are encoded in parallel batches (tiktoken releases the GIL), which
    is far faster than one-at-a-time for short docs like TinyStories.
    """
    num_threads = num_threads or (os.cpu_count() or 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    texts = iter_texts(spec, streaming=streaming, limit=limit)
    n_docs = n_tokens = 0
    start = time.time()
    with open(output, "wb") as f:
        bar = tqdm()
        for batch in _batched(texts, batch_size):
            encoded = enc.encode_ordinary_batch(batch, num_threads=num_threads)
            flat: list[int] = []
            for ids in encoded:
                flat.extend(ids)
                flat.append(eos_id)
            np.asarray(flat, dtype=DTYPE).tofile(f)
            n_docs += len(batch)
            n_tokens += len(flat)
            rate = n_tokens / (time.time() - start)
            bar.set_description(
                f"{output.name}: {n_docs:,} docs | {n_tokens:,} tokens | {rate:,.0f} tok/s"
            )
            bar.update(len(batch))
        bar.close()
    print(
        f"done: {n_docs:,} docs, {n_tokens:,} tokens -> {output} "
        f"({output.stat().st_size / 1e6:.1f} MB)"
    )
    return n_docs, n_tokens


def run_config(cfg: dict, enc, eos_id: int) -> None:
    """Process every dataset listed in a preprocess recipe.

    Each entry picks its own `split` (default train), so validation bins are
    just ordinary entries pointing at a validation split, e.g.:
        - {preset: tinystories, output: tinystories.bin}
        - {preset: tinystories, output: tinystories.val.bin, split: validation}
    Entries whose split doesn't exist (e.g. validation for a train-only dataset)
    are skipped rather than erroring.
    """
    output_dir = Path(cfg.get("output_dir", ""))
    streaming = cfg.get("streaming", False)
    batch_size = cfg.get("batch_size", BATCH_SIZE)
    num_threads = cfg.get("num_threads")
    entries = cfg["datasets"]
    for i, entry in enumerate(entries, 1):
        spec = spec_from_entry(entry)
        if "output" not in entry:
            raise SystemExit(f"dataset entry needs 'output': {entry}")
        output = output_dir / entry["output"]
        if split_unavailable(spec):
            print(
                f"[{i}/{len(entries)}] skip {spec.path}: no '{spec.split}' split",
                flush=True,
            )
            continue
        limit = entry.get("limit")
        # Per-entry `streaming` overrides the file default, e.g. to stream a small
        # slice of a huge dataset instead of downloading all of it.
        entry_streaming = entry.get("streaming", streaming)
        print(f"[{i}/{len(entries)}] {spec.path} ({spec.split}) -> {output}", flush=True)
        process(
            spec,
            output,
            enc,
            eos_id,
            streaming=entry_streaming,
            limit=limit,
            batch_size=batch_size,
            num_threads=num_threads,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", type=str, default="weights/tokenizer.json")
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="preprocess recipe (YAML)"
    )
    # Single-dataset (ad-hoc) mode; ignored when --config is given.
    parser.add_argument("-o", "--output", type=str, default=None, help="output .bin path")
    parser.add_argument("-p", "--preset", type=str, default=None)
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument(
        "-s", "--split", type=str, default=None, help="override the spec's split"
    )
    parser.add_argument("--limit", type=int, default=None)
    # Default is non-streaming: download once to the HF cache, then iterate from
    # local arrow (much faster). Pass --streaming for datasets too big for disk.
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="docs per encode batch"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="tiktoken encode threads (default: os.cpu_count())",
    )
    args = parser.parse_args()

    enc, eos_id = load_enc(args.tokenizer)

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if "tokenizer" in cfg:
            enc, eos_id = load_enc(cfg["tokenizer"])
        run_config(cfg, enc, eos_id)
        return

    if not args.output:
        raise SystemExit("either --config, or --output with --preset/--dataset")
    spec = resolve_spec(args.preset, args.dataset)
    if args.split is not None:
        spec.split = args.split
    process(
        spec,
        Path(args.output),
        enc,
        eos_id,
        streaming=args.streaming,
        limit=args.limit,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
