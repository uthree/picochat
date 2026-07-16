"""Train a BPE tokenizer from a YAML recipe.

The tokenizer recipe is independent from the pretraining stage configs (it has
its own vocab size and data mixture). See configs/tok/*.yml.

    python scripts/tok_train.py --config configs/tok/en_ja.yml
"""

import argparse
from pathlib import Path

import yaml

from picochat.dataset import TEXT_PRESETS, DatasetSpec, Mixture, iter_mixture
from picochat.tokenizer import SPECIAL_TOKENS, train_tokenizer


def spec_from_entry(entry: dict) -> DatasetSpec:
    """Resolve one `data:` entry into a DatasetSpec.

    Either {preset: <name>} referencing picochat.dataset, or an inline
    {path, name, split, text_key}.
    """
    if "preset" in entry:
        name = entry["preset"]
        if name not in TEXT_PRESETS:
            raise SystemExit(
                f"unknown preset '{name}'. choices: {', '.join(TEXT_PRESETS)}"
            )
        return TEXT_PRESETS[name]
    if "path" in entry:
        return DatasetSpec(
            path=entry["path"],
            name=entry.get("name"),
            split=entry.get("split", "train"),
            text_key=entry.get("text_key", "text"),
        )
    raise SystemExit(f"data entry needs 'preset' or 'path': {entry}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="tokenizer recipe (YAML)",
        default="configs/tok/en_ja.yml",
    )
    p.add_argument(
        "-o", "--output", type=str, default=None, help="override config's output path"
    )
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vocab_size = cfg["vocab_size"]
    total_chars = cfg.get("total_chars", 500_000_000)
    streaming = cfg.get("streaming", True)
    output = Path(args.output or cfg.get("output", "weights/tokenizer.json"))
    output.parent.mkdir(parents=True, exist_ok=True)

    entries = cfg["data"]
    specs = [spec_from_entry(e) for e in entries]
    raw_weights = [float(e.get("weight", 1.0)) for e in entries]
    total = sum(raw_weights)
    # Normalize so weights are fractions of total_chars (byte balancing).
    weights = [w / total for w in raw_weights]
    mixture = Mixture(specs=specs, weights=weights)

    texts = iter_mixture(mixture, total_chars=total_chars, streaming=streaming)
    print("training tokenizer ...")
    train_tokenizer(
        texts,
        vocab_size=vocab_size,
        save_as=output,
        special_tokens=SPECIAL_TOKENS,
    )
    print(f"saved tokenizer to {output} (vocab_size={vocab_size})")


if __name__ == "__main__":
    main()
