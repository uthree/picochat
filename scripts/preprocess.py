"""HF データセットをトークナイズし、packing 済みの flat uint16 バイナリに変換する。

各文書を encode し末尾に <eos> を付けて連結し、1本の連続したトークン列として
.bin に書き出す。padding は一切入れず、学習側（PackedDataset）が読み出し時に
block_size+1 の窓でスライスする。vocab_size <= 65535 なら uint16 で収まる。
"""

import argparse
import time
from pathlib import Path

import numpy as np

from picochat.data.sources import iter_texts, resolve_spec
from picochat.tokenizer import load_tokenizer

DTYPE = np.uint16  # vocab_size <= 65535 を前提
EOS_TOKEN = "</s>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--tokenizer", type=str, default="weights/tokenizer.json"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="出力 .bin パス"
    )
    parser.add_argument("-p", "--preset", type=str, default=None)
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument(
        "-s", "--split", type=str, default=None, help="spec の split を上書き"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument(
        "--log-every", type=int, default=10000, help="N 文書ごとに進捗を表示"
    )
    args = parser.parse_args()

    enc = load_tokenizer(args.tokenizer)
    eos_id = enc._special_tokens[EOS_TOKEN]
    assert enc.n_vocab <= np.iinfo(DTYPE).max + 1, (
        f"vocab {enc.n_vocab} は {DTYPE} に収まらない"
    )

    spec = resolve_spec(args.preset, args.dataset)
    if args.split is not None:
        spec.split = args.split

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    texts = iter_texts(spec, streaming=not args.no_streaming, limit=args.limit)
    n_docs = 0
    n_tokens = 0
    start = time.time()
    with open(output, "wb") as f:
        for text in texts:
            ids = enc.encode_ordinary(text)
            ids.append(eos_id)
            np.asarray(ids, dtype=DTYPE).tofile(f)
            n_docs += 1
            n_tokens += len(ids)
            if n_docs % args.log_every == 0:
                rate = n_tokens / (time.time() - start)
                print(
                    f"{n_docs:,} docs | {n_tokens:,} tokens | {rate:,.0f} tok/s",
                    flush=True,
                )

    print(
        f"done: {n_docs:,} docs, {n_tokens:,} tokens -> {output} "
        f"({output.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
