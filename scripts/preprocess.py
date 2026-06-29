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
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from picochat.data.pretrain import DTYPE  # uint32; shared with the reader
from picochat.data.sources import PRESETS, DatasetSpec, iter_texts, resolve_spec
from picochat.tokenizer import load_tokenizer

EOS_TOKEN = "</s>"


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
    if "split" in entry:
        spec.split = entry["split"]
    return spec


def process(
    spec: DatasetSpec,
    output: Path,
    enc,
    eos_id: int,
    streaming: bool = True,
    limit: int | None = None,
) -> tuple[int, int]:
    """Encode every document of `spec` into `output`. Returns (n_docs, n_tokens)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    texts = iter_texts(spec, streaming=streaming, limit=limit)
    n_docs = n_tokens = 0
    start = time.time()
    with open(output, "wb") as f:
        bar = tqdm()
        for text in texts:
            ids = enc.encode_ordinary(text)
            ids.append(eos_id)
            np.asarray(ids, dtype=DTYPE).tofile(f)
            n_docs += 1
            n_tokens += len(ids)
            rate = n_tokens / (time.time() - start)
            bar.set_description(
                f"{output.name}: {n_docs:,} docs | {n_tokens:,} tokens | {rate:,.0f} tok/s"
            )
            bar.update()
        bar.close()
    print(
        f"done: {n_docs:,} docs, {n_tokens:,} tokens -> {output} "
        f"({output.stat().st_size / 1e6:.1f} MB)"
    )
    return n_docs, n_tokens


def run_config(cfg: dict, enc, eos_id: int) -> None:
    """Process every dataset listed in a preprocess recipe."""
    output_dir = Path(cfg.get("output_dir", ""))
    streaming = cfg.get("streaming", True)
    entries = cfg["datasets"]
    for i, entry in enumerate(entries, 1):
        spec = spec_from_entry(entry)
        if "output" not in entry:
            raise SystemExit(f"dataset entry needs 'output': {entry}")
        output = output_dir / entry["output"]
        limit = entry.get("limit")
        print(f"[{i}/{len(entries)}] {spec.path} ({spec.split}) -> {output}", flush=True)
        process(spec, output, enc, eos_id, streaming=streaming, limit=limit)


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
    parser.add_argument("--no-streaming", action="store_true")
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
        streaming=not args.no_streaming,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
