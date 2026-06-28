import argparse
from pathlib import Path

from picochat.data.sources import RECIPES, iter_mixture, iter_texts, resolve_spec
from picochat.tokenizer import train_tokenizer

NUM_RESERVED_SPECIAL_TOKENS = 16
SPECIAL_TOKENS = [
    "<pad>",  # padding
    "<mask>",  # mask (not used in causal language model)
    "<unk>",  # unknown word (Not used with the BPE tokenizer.)
    "<sep>",  # separator for multiple sentences.
    "<think>",  # start thinking (for Chain of Tought)
    "</think>",  # Stop thinking
    "<s>",  # start decoding
    "</s>",  # stop decoding
] + [f"<reserved_token_{n}>" for n in range(NUM_RESERVED_SPECIAL_TOKENS)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocab-size", type=int, default=64000)
    parser.add_argument("-o", "--output", type=str, default="weights/tokenizer.json")
    parser.add_argument(
        "-r",
        "--recipe",
        type=str,
        default=None,
        help=f"name of a multi-source mixing recipe {list(RECIPES)}",
    )
    parser.add_argument(
        "--total-chars",
        type=int,
        default=500_000_000,
        help="total characters to read when using a recipe (byte-balanced across languages)",
    )
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        default=None,
        help="preset name from picochat.data.sources",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help='HF dataset in "path[:name[:split[:text_key]]]" form',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="number of leading texts to train on (for sanity checks)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="disable streaming (download everything before processing)",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.recipe is not None:
        if args.recipe not in RECIPES:
            raise SystemExit(
                f"unknown recipe '{args.recipe}'. choices: {list(RECIPES)}"
            )
        texts = iter_mixture(
            RECIPES[args.recipe],
            total_chars=args.total_chars,
            streaming=not args.no_streaming,
        )
    else:
        spec = resolve_spec(args.preset, args.dataset)
        texts = iter_texts(spec, streaming=not args.no_streaming, limit=args.limit)
    train_tokenizer(
        texts,
        vocab_size=args.vocab_size,
        save_as=output,
        special_tokens=SPECIAL_TOKENS,
    )
    print(f"saved tokenizer to {output} (vocab_size={args.vocab_size})")


if __name__ == "__main__":
    main()
