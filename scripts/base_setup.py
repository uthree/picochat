"""Tokenize HF datasets and convert them into packed, sharded token binaries.

Each document is encoded and wrapped in <|begin_of_text|>...<|end_of_text|>,
then whole documents are packed into fixed-length rows of block_size+1 tokens
(MosaicBERT-style sequence packing, greedy best-fit -- see
picochat.dataloader.pack_docs; documents longer than one row are split into
row-sized chunks, each continuation prefixed with <|begin_of_text|>). The rows
are split across shard files (00000.bin, 00001.bin, ...) under one output
directory per dataset, so no single file grows with the corpus, plus a
meta.json recording the block_size; the training side (PackedDataset) reads
one row per sample and rejects a mismatched block_size. Row leftovers are
padded with <|pad|>, whose targets the training loss already ignores. Packing
happens one encode batch at a time, so only one batch is ever held in memory.
Tokens are stored as uint32 (DTYPE), which fits vocab up to ~4.29B.

`block_size` must match the training config's data.block_size, so it is a
required setting here (config `block_size:` / ad-hoc `--block-size`).

Two ways to run:

  # one dataset, ad-hoc (output is a shard directory)
  python scripts/base_setup.py -p tinystories -o data/tinystories --block-size 512

  # many datasets from a recipe (configs/base_setup/*.yml)
  python scripts/base_setup.py --config configs/base_setup/base.yml
"""

import argparse
import os
import time
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml
from datasets import get_dataset_split_names, load_dataset_builder
from tqdm import tqdm

from picochat.dataloader import (  # DTYPE: uint32; shared with the reader
    DEFAULT_SHARD_TOKENS,
    DTYPE,
    ShardWriter,
    pack_docs,
    write_meta,
)
from picochat.dataset import (
    DatasetSpec,
    holdout_splits,
    iter_texts,
    resolve_text_spec,
    spec_from_entry,
)
from picochat.tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, load_tokenizer

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
    """Load the tokenizer and return (encoding, bos_id, eos_id, pad_id),
    checking vocab fits DTYPE."""
    enc = load_tokenizer(tokenizer_path)
    bos_id = enc._special_tokens[BOS_TOKEN]
    eos_id = enc._special_tokens[EOS_TOKEN]
    pad_id = enc._special_tokens[PAD_TOKEN]
    assert enc.n_vocab <= np.iinfo(DTYPE).max + 1, (
        f"vocab {enc.n_vocab} does not fit in {DTYPE}"
    )
    return enc, bos_id, eos_id, pad_id


def _split_example_count(spec: DatasetSpec) -> int:
    """Look up `spec.split`'s row count from the Hub's dataset info (metadata
    only, no data download) -- what holdout_splits needs to turn a
    `val_fraction` into an exact absolute-index slice."""
    try:
        info = load_dataset_builder(spec.path, spec.name).info
        return info.splits[spec.split].num_examples
    except Exception as e:
        raise SystemExit(
            f"couldn't resolve example count for {spec.path} (name={spec.name!r}, "
            f"split={spec.split!r}) to compute val_fraction: {type(e).__name__}: {e}"
        ) from e


def expand_val_fraction(entries: list[dict]) -> list[dict]:
    """Expand a `val_fraction`/`val_output` entry into a plain train + val pair
    (see picochat.dataset.holdout_splits), so the rest of the pipeline
    never has to know about held-out splits. For datasets with no dedicated
    validation split (e.g. Wikipedia, cosmopedia -- only "train"), this is
    how a val set is carved out instead of falling back to a different
    dataset's split, which would leave that dataset's slice of the training
    mixture unrepresented in validation loss:
        - {preset: wikipedia-en, output: wikipedia-en, val_output: wikipedia-en.val, val_fraction: 0.002}
    Entries without `val_fraction` pass through unchanged.
    """
    expanded: list[dict] = []
    for entry in entries:
        if "val_fraction" not in entry:
            expanded.append(entry)
            continue
        if "val_output" not in entry:
            raise SystemExit(f"'val_fraction' entry needs 'val_output': {entry}")
        base = {
            k: v for k, v in entry.items() if k not in ("val_fraction", "val_output")
        }
        spec = spec_from_entry(entry)
        train_split, val_split = holdout_splits(
            spec.split, entry["val_fraction"], _split_example_count(spec)
        )
        expanded.append({**base, "split": train_split})
        expanded.append({**base, "output": entry["val_output"], "split": val_split})
    return expanded


def process(
    spec: DatasetSpec,
    output: Path,
    enc,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    block_size: int,
    streaming: bool = False,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
    num_threads: int | None = None,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
) -> tuple[int, int]:
    """Encode every document of `spec` and pack them into fixed-length rows of
    block_size+1 tokens in shard files under the `output` directory (see
    pack_docs). Returns (n_docs, n_tokens) where n_tokens counts the real
    (non-padding) tokens written.

    Documents are encoded in parallel batches (tiktoken releases the GIL), which
    is far faster than one-at-a-time for short docs like TinyStories. Each batch
    is packed and flushed to the ShardWriter as soon as it's encoded, so peak
    memory stays O(batch) no matter how large the corpus is; packing efficiency
    only pays for it at the last few rows of each batch.
    """
    row_len = block_size + 1
    # Rows must never straddle a shard boundary, so round the shard capacity
    # down to a whole number of rows.
    shard_tokens = max(row_len, shard_tokens // row_len * row_len)
    num_threads = num_threads or (os.cpu_count() or 8)
    texts = iter_texts(spec, streaming=streaming, limit=limit)
    n_docs = n_tokens = n_rows = 0
    start = time.time()
    writer = ShardWriter(output, shard_tokens)
    write_meta(output, block_size)
    bar = tqdm()
    try:
        for batch in _batched(texts, batch_size):
            encoded = enc.encode_ordinary_batch(batch, num_threads=num_threads)
            docs = [[bos_id, *ids, eos_id] for ids in encoded]
            rows = pack_docs(docs, block_size, pad_id, bos_id)
            writer.write(rows.reshape(-1))
            n_docs += len(batch)
            n_rows += len(rows)
            n_tokens += int((rows != pad_id).sum())
            rate = n_tokens / (time.time() - start)
            bar.set_description(
                f"{output.name}: {n_docs:,} docs | {n_tokens:,} tokens | {rate:,.0f} tok/s"
            )
            bar.update(len(batch))
    finally:
        bar.close()
        writer.close()
    size_mb = sum(f.stat().st_size for f in output.glob("*.bin")) / 1e6
    efficiency = n_tokens / max(1, n_rows * row_len)  # real tokens vs padding
    print(
        f"done: {n_docs:,} docs, {n_tokens:,} tokens -> {output}/ "
        f"({n_rows:,} packed rows, {efficiency:.1%} packing efficiency, "
        f"{writer.n_shards} shard(s), {size_mb:.1f} MB)"
    )
    return n_docs, n_tokens


def run_config(cfg: dict, enc, bos_id: int, eos_id: int, pad_id: int) -> None:
    """Process every dataset listed in a preprocess recipe.

    Each entry picks its own `split` (default train), so validation bins are
    just ordinary entries pointing at a validation split, e.g.:
        - {preset: tinystories, output: tinystories.bin}
        - {preset: tinystories, output: tinystories.val.bin, split: validation}
    For datasets with no dedicated validation split, `val_fraction` carves one
    out of `split` instead (see expand_val_fraction). Every dataset/split is
    validated up front; an invalid one stops the script.
    """
    output_dir = Path(cfg.get("output_dir", ""))
    streaming = cfg.get("streaming", False)
    batch_size = cfg.get("batch_size", BATCH_SIZE)
    num_threads = cfg.get("num_threads")
    shard_tokens = cfg.get("shard_tokens", DEFAULT_SHARD_TOKENS)
    if "block_size" not in cfg:
        raise SystemExit(
            "config needs 'block_size' (rows are packed to block_size+1 tokens; "
            "must match the training config's data.block_size)"
        )
    block_size = cfg["block_size"]
    entries = expand_val_fraction(cfg["datasets"])
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
            pad_id,
            block_size,
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
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="pack rows of block_size+1 tokens; must match the training "
        "config's data.block_size (required unless --config supplies it)",
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
        enc, bos_id, eos_id, pad_id = load_enc(cfg["tokenizer"])
        run_config(cfg, enc, bos_id, eos_id, pad_id)
        return

    if not args.output:
        raise SystemExit("either --config, or --output with --preset/--dataset")
    if args.block_size is None:
        raise SystemExit(
            "--block-size is required (must match the training config's "
            "data.block_size)"
        )
    enc, bos_id, eos_id, pad_id = load_enc(DEFAULT_TOKENIZER)
    spec = resolve_text_spec(args.preset, args.dataset)
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
        pad_id,
        args.block_size,
        streaming=args.streaming,
        limit=args.limit,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        shard_tokens=args.shard_tokens,
    )


if __name__ == "__main__":
    main()
