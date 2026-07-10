"""Tokenize HF datasets and convert them into packed, sharded token binaries.

Each document is encoded and wrapped in <|begin_of_text|>...<|end_of_text|>,
and everything is concatenated into a continuous token stream split across
shard files (00000.bin, 00001.bin, ...) under one output directory per
dataset, so no single file grows with the corpus:
<|begin_of_text|>doc1<|end_of_text|><|begin_of_text|>doc2<|end_of_text|>...
Only one encode batch is ever held in memory. No padding is added; the
training side (PackedDataset) slices a block_size+1 window at read time.
Tokens are stored as uint32 (DTYPE), which fits vocab up to ~4.29B.

Two ways to run:

  # one dataset, ad-hoc (output is a shard directory)
  python scripts/base_setup.py -p tinystories -o data/tinystories

  # many datasets from a recipe (configs/base_setup/*.yml)
  python scripts/base_setup.py --config configs/base_setup/stage1.yml
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
from datasets import get_dataset_split_names
from tqdm import tqdm

from picochat.data.pretrain import (  # DTYPE: uint32; shared with the reader
    DEFAULT_SHARD_TOKENS,
    DTYPE,
    PRESETS,
    DatasetSpec,
    ShardWriter,
    iter_texts,
    resolve_spec,
)
from picochat.tokenizer import BOS_TOKEN, EOS_TOKEN, load_tokenizer

# Used by the ad-hoc single-dataset mode; config mode reads the path from the
# recipe's `tokenizer:` field instead.
DEFAULT_TOKENIZER = "weights/tokenizer.json"
# Encoding throughput is dominated by tiktoken. Encoding one doc at a time has
# heavy per-call overhead, so we feed it large batches: encode_ordinary_batch
# tokenizes them in parallel across Rust threads (GIL released).
BATCH_SIZE = 1024


def _batched(it: Iterator[str], n: int) -> Iterator[list[str]]:
    while batch := list(islice(it, n)):
        yield batch


def validate_specs(specs: list[DatasetSpec]) -> list[str]:
    """Return a problem message for every spec whose dataset or split is invalid.

    Run before any processing so a typo'd path/config or a missing split (e.g. a
    validation entry for a train-only dataset like wikipedia) stops the script
    up front instead of failing halfway through. Split listings are metadata-only
    (no data download) and cached per (path, name).
    """
    problems: list[str] = []
    splits_cache: dict[tuple[str, str | None], list[str] | None] = {}
    for spec in specs:
        key = (spec.path, spec.name)
        if key not in splits_cache:
            try:
                splits_cache[key] = get_dataset_split_names(spec.path, spec.name)
            except Exception as e:
                splits_cache[key] = None
                problems.append(
                    f"{spec.path} (name={spec.name!r}): cannot load dataset "
                    f"({type(e).__name__}: {e})"
                )
        available = splits_cache[key]
        if available is None:
            continue  # dataset-level failure already reported
        base = spec.split.split("[")[0]  # ignore slicing like "train[:1%]"
        if base not in available:
            problems.append(
                f"{spec.path} (name={spec.name!r}): split {spec.split!r} not found; "
                f"available: {available}"
            )
    return problems


def load_enc(tokenizer_path: str):
    """Load the tokenizer and return (encoding, bos_id, eos_id), checking vocab
    fits DTYPE."""
    enc = load_tokenizer(tokenizer_path)
    bos_id = enc._special_tokens[BOS_TOKEN]
    eos_id = enc._special_tokens[EOS_TOKEN]
    assert enc.n_vocab <= np.iinfo(DTYPE).max + 1, (
        f"vocab {enc.n_vocab} does not fit in {DTYPE}"
    )
    return enc, bos_id, eos_id


def spec_from_entry(entry: dict) -> DatasetSpec:
    """Resolve one `datasets:` entry into a DatasetSpec.

    Either {preset: <name>} referencing picochat.data.pretrain, or an inline
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
    bos_id: int,
    eos_id: int,
    streaming: bool = False,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
    num_threads: int | None = None,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
) -> tuple[int, int]:
    """Encode every document of `spec` into shard files under the `output`
    directory. Returns (n_docs, n_tokens).

    Documents are encoded in parallel batches (tiktoken releases the GIL), which
    is far faster than one-at-a-time for short docs like TinyStories. Each batch
    is flushed to the ShardWriter as soon as it's encoded, so peak memory stays
    O(batch) no matter how large the corpus is.
    """
    num_threads = num_threads or (os.cpu_count() or 8)
    texts = iter_texts(spec, streaming=streaming, limit=limit)
    n_docs = n_tokens = 0
    start = time.time()
    writer = ShardWriter(output, shard_tokens)
    bos = np.asarray([bos_id], dtype=DTYPE)
    eos = np.asarray([eos_id], dtype=DTYPE)
    bar = tqdm()
    try:
        for batch in _batched(texts, batch_size):
            encoded = enc.encode_ordinary_batch(batch, num_threads=num_threads)
            parts: list[np.ndarray] = []
            for ids in encoded:
                parts.append(bos)
                parts.append(np.asarray(ids, dtype=DTYPE))
                parts.append(eos)
            tokens = np.concatenate(parts)
            writer.write(tokens)
            n_docs += len(batch)
            n_tokens += int(tokens.size)
            rate = n_tokens / (time.time() - start)
            bar.set_description(
                f"{output.name}: {n_docs:,} docs | {n_tokens:,} tokens | {rate:,.0f} tok/s"
            )
            bar.update(len(batch))
    finally:
        bar.close()
        writer.close()
    size_mb = sum(f.stat().st_size for f in output.glob("*.bin")) / 1e6
    print(
        f"done: {n_docs:,} docs, {n_tokens:,} tokens -> {output}/ "
        f"({writer.n_shards} shard(s), {size_mb:.1f} MB)"
    )
    return n_docs, n_tokens


def run_config(cfg: dict, enc, bos_id: int, eos_id: int) -> None:
    """Process every dataset listed in a preprocess recipe.

    Each entry picks its own `split` (default train), so validation bins are
    just ordinary entries pointing at a validation split, e.g.:
        - {preset: tinystories, output: tinystories.bin}
        - {preset: tinystories, output: tinystories.val.bin, split: validation}
    Every dataset/split is validated up front; an invalid one stops the script.
    """
    output_dir = Path(cfg.get("output_dir", ""))
    streaming = cfg.get("streaming", False)
    batch_size = cfg.get("batch_size", BATCH_SIZE)
    num_threads = cfg.get("num_threads")
    shard_tokens = cfg.get("shard_tokens", DEFAULT_SHARD_TOKENS)
    entries = cfg["datasets"]
    for entry in entries:
        if "output" not in entry:
            raise SystemExit(f"dataset entry needs 'output': {entry}")
    specs = [spec_from_entry(entry) for entry in entries]

    print(f"validating {len(specs)} dataset(s)/split(s)...", flush=True)
    problems = validate_specs(specs)
    if problems:
        raise SystemExit(
            "invalid dataset/split in config (nothing processed):\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

    for i, (entry, spec) in enumerate(zip(entries, specs), 1):
        output = output_dir / entry["output"]
        limit = entry.get("limit")
        # Per-entry `streaming` overrides the file default, e.g. to stream a small
        # slice of a huge dataset instead of downloading all of it.
        entry_streaming = entry.get("streaming", streaming)
        print(
            f"[{i}/{len(entries)}] {spec.path} ({spec.split}) -> {output}", flush=True
        )
        process(
            spec,
            output,
            enc,
            bos_id,
            eos_id,
            streaming=entry_streaming,
            limit=limit,
            batch_size=batch_size,
            num_threads=num_threads,
            shard_tokens=entry.get("shard_tokens", shard_tokens),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="preprocess recipe (YAML)"
    )
    # Single-dataset (ad-hoc) mode; ignored when --config is given.
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="output shard directory"
    )
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=DEFAULT_SHARD_TOKENS,
        help="max tokens per shard file (default: 2**28 = 1 GiB of uint32)",
    )
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

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if "tokenizer" not in cfg:
            raise SystemExit(
                f"{args.config} needs a 'tokenizer:' field (path to tokenizer.json)"
            )
        enc, bos_id, eos_id = load_enc(cfg["tokenizer"])
        run_config(cfg, enc, bos_id, eos_id)
        return

    if not args.output:
        raise SystemExit("either --config, or --output with --preset/--dataset")
    enc, bos_id, eos_id = load_enc(DEFAULT_TOKENIZER)
    spec = resolve_spec(args.preset, args.dataset)
    if args.split is not None:
        spec.split = args.split
    problems = validate_specs([spec])
    if problems:
        raise SystemExit(
            "invalid dataset/split:\n" + "\n".join(f"  - {p}" for p in problems)
        )
    process(
        spec,
        Path(args.output),
        enc,
        bos_id,
        eos_id,
        streaming=args.streaming,
        limit=args.limit,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        shard_tokens=args.shard_tokens,
    )


if __name__ == "__main__":
    main()
