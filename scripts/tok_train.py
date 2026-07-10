"""Train a BPE tokenizer from a YAML recipe.

The tokenizer recipe is independent from the pretraining stage configs (it has
its own vocab size and data mixture). See configs/tok/*.yml.

    python scripts/tok_train.py --config configs/tok/en_ja.yml
"""

import argparse
from pathlib import Path

import yaml

from picochat.data.pretrain import PRESETS, DatasetSpec, Mixture, iter_mixture
from picochat.tokenizer import train_tokenizer

# Two reserved slots were repurposed for the ChatML turn delimiters, keeping
# the total (and thus the vocab size) unchanged; an existing tokenizer.json
# can be migrated in place with scripts/tok_patch_chatml.py instead of
# retraining.
NUM_RESERVED_SPECIAL_TOKENS = 14
SPECIAL_TOKENS = [
    "<pad>",  # padding
    "<mask>",  # mask (not used in causal language model)
    "<unk>",  # unknown word (Not used with the BPE tokenizer.)
    "<sep>",  # separator for multiple sentences.
    "<think>",  # start thinking (for Chain of Tought)
    "</think>",  # Stop thinking
    "<s>",  # start of a document (pretraining) / conversation (SFT)
    "</s>",  # end of a document / conversation
    "<|im_start|>",  # ChatML: start of a turn (followed by "{role}\n")
    "<|im_end|>",  # ChatML: end of a turn (the chat stop token)
] + [f"<reserved_token_{n}>" for n in range(NUM_RESERVED_SPECIAL_TOKENS)]


def spec_from_entry(entry: dict) -> DatasetSpec:
    """Resolve one `data:` entry into a DatasetSpec.

    Either {preset: <name>} referencing picochat.data.pretrain, or an inline
    {path, name, split, text_key}.
    """
    if "preset" in entry:
        name = entry["preset"]
        if name not in PRESETS:
            raise SystemExit(f"unknown preset '{name}'. choices: {', '.join(PRESETS)}")
        return PRESETS[name]
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
