"""Measure tokenizer efficiency (bytes/token) per dataset.

Scanning a full dataset just to check how well the tokenizer compresses it is
slow, so this only reads the first N documents of each source listed in a
tok_train.py recipe (configs/tok/*.yml), encodes them, and reports
bytes/token per dataset -- a higher number means the tokenizer packs that
dataset's text into fewer tokens (denser fit for that language/domain).

    python scripts/tok_eval.py                              # every configs/tok/*.yml
    python scripts/tok_eval.py --config configs/tok/en_ja.yml
    python scripts/tok_eval.py --config configs/tok/en_ja.yml -n 20000
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Iterator

from tqdm import tqdm
import yaml

from picochat.dataset import DatasetSpec, iter_texts, spec_from_entry
from picochat.tokenizer import load_tokenizer

BATCH_SIZE = 1024
DEFAULT_LIMIT = 10000


def _batched(texts: Iterator[str], n: int) -> Iterator[list[str]]:
    batch = []
    for text in texts:
        batch.append(text)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def eval_dataset(
    spec: DatasetSpec, enc, limit: int, batch_size: int, num_threads: int
) -> tuple[int, int, int]:
    """Encode the first `limit` docs of `spec`. Returns (n_docs, n_bytes, n_tokens).

    Streams so only the first `limit` documents are ever downloaded, not the
    whole dataset.
    """
    label = spec.path + (f"/{spec.name}" if spec.name else "")
    texts = iter_texts(spec, streaming=True, limit=limit)
    n_docs = n_bytes = n_tokens = 0
    bar = tqdm(total=limit, desc=label, unit="doc")
    for batch in _batched(texts, batch_size):
        encoded = enc.encode_ordinary_batch(batch, num_threads=num_threads)
        n_docs += len(batch)
        n_bytes += sum(len(text.encode("utf-8")) for text in batch)
        n_tokens += sum(len(ids) for ids in encoded)
        bar.update(len(batch))
    bar.close()
    return n_docs, n_bytes, n_tokens


def eval_config(
    path: Path,
    limit: int,
    tokenizer_override: str | None,
    batch_size: int,
    num_threads: int,
) -> None:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    tokenizer_path = tokenizer_override or cfg.get("output", "weights/tokenizer.json")
    enc = load_tokenizer(tokenizer_path)
    specs = [spec_from_entry(e) for e in cfg["data"]]

    print(
        f"\n=== {path} (tokenizer={tokenizer_path}, vocab={enc.n_vocab}, n={limit}) ==="
    )
    rows = []
    for spec in specs:
        n_docs, n_bytes, n_tokens = eval_dataset(
            spec, enc, limit, batch_size, num_threads
        )
        label = spec.path + (f"/{spec.name}" if spec.name else "")
        rows.append((label, n_docs, n_bytes, n_tokens))

    name_w = max(len(label) for label, *_ in rows)
    total_bytes = total_tokens = 0
    for label, n_docs, n_bytes, n_tokens in rows:
        bytes_per_tok = n_bytes / n_tokens if n_tokens else float("nan")
        print(
            f"  {label:<{name_w}}  docs={n_docs:>6,}  bytes={n_bytes:>10,}  "
            f"tokens={n_tokens:>10,}  bytes/tok={bytes_per_tok:6.3f}"
        )
        total_bytes += n_bytes
        total_tokens += n_tokens
    overall = total_bytes / total_tokens if total_tokens else float("nan")
    print(
        f"  {'(overall)':<{name_w}}  {'':<12}bytes={total_bytes:>10,}  "
        f"tokens={total_tokens:>10,}  bytes/tok={overall:6.3f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="single tok recipe (YAML); default: every configs/tok/*.yml",
    )
    p.add_argument(
        "-n",
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"docs per dataset, from the start (default: {DEFAULT_LIMIT})",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="override the config's tokenizer path",
    )
    p.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="docs per encode batch"
    )
    p.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="tiktoken encode threads (default: os.cpu_count())",
    )
    args = p.parse_args()

    if args.config:
        paths = [Path(args.config)]
    else:
        paths = sorted(Path(f) for f in glob.glob("configs/tok/*.yml"))
        if not paths:
            raise SystemExit("no configs found under configs/tok/*.yml")

    num_threads = args.num_threads or (os.cpu_count() or 8)
    for path in paths:
        eval_config(path, args.limit, args.tokenizer, args.batch_size, num_threads)


if __name__ == "__main__":
    main()
