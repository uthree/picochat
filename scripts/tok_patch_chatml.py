"""Patch an existing tokenizer.json in place for ChatML, without retraining.

scripts/tok_train.py used to reserve 16 unused special-token slots
(<reserved_token_0..15>); the ChatML turn delimiters now occupy the first two.
This script renames <reserved_token_0> -> <|im_start|> and
<reserved_token_1> -> <|im_end|>, then renumbers the remaining reserved
tokens down by two, so the patched file is byte-for-byte equivalent to what a
fresh tok_train.py run would produce -- same ids, same vocab size, no BPE
retraining, and existing pretraining .bin shards stay valid.

    python scripts/tok_patch_chatml.py weights/tokenizer.json
"""

import argparse
import json


def patch(path: str) -> None:
    with open(path) as f:
        data = json.load(f)
    special = data["special_tokens"]

    if "<|im_start|>" in special and "<|im_end|>" in special:
        print(f"{path}: already patched, nothing to do")
        return
    for required in ("<reserved_token_0>", "<reserved_token_1>"):
        if required not in special:
            raise SystemExit(
                f"{path}: no {required} to repurpose -- this tokenizer wasn't "
                "trained with tok_train.py's reserved slots; retrain instead."
            )

    special["<|im_start|>"] = special.pop("<reserved_token_0>")
    special["<|im_end|>"] = special.pop("<reserved_token_1>")
    # Renumber the rest so the names match a fresh tok_train.py run.
    n = 2
    while (old := f"<reserved_token_{n}>") in special:
        special[f"<reserved_token_{n - 2}>"] = special.pop(old)
        n += 1

    with open(path, "w") as f:
        json.dump(data, f)
    print(
        f"{path}: patched -- <|im_start|>={special['<|im_start|>']}, "
        f"<|im_end|>={special['<|im_end|>']}, {n - 2} reserved slot(s) left"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer", help="path to a tokenizer.json to patch in place")
    args = p.parse_args()
    patch(args.tokenizer)


if __name__ == "__main__":
    main()
