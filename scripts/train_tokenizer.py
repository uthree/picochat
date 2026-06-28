import argparse

from picochat.tokenizer import load_tokenizer, train_tokenizer

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
    parser.add_argument("-v", "--vocab-size", type=int, default=32000)
    parser.add_argument("-o", "--output", type=str, default="weights/tokenizer.json")
    # TODO: load dataset and train tokenizer


if __name__ == "__main__":
    main()
