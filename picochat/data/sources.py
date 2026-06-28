"""HuggingFace のテキストデータセットを共通の文字列イテレータに変換する。

トークナイザ学習（scripts/train_tokenizer.py）と前処理（scripts/preprocess.py）の
両方が「データセットからテキストを1件ずつ流す」処理を必要とするので、ここに集約する。

streaming=True なら巨大なデータセットでもシャードを遅延ダウンロードするため、
limit を併用すれば先頭 N 件だけで動作確認できる。
"""

from dataclasses import dataclass, field
from typing import Callable, Iterator

from datasets import load_dataset


@dataclass
class DatasetSpec:
    path: str  # HF Hub のリポジトリ ID 例: "roneneldan/TinyStories"
    name: str | None = None  # config / subset 名 例: "20231101.en"
    split: str = "train"
    text_key: str = "text"  # テキストが入っているカラム名
    # 複数カラムを1つのテキストに整形したい場合（例: GSM8K の question + answer）
    format: Callable[[dict], str] | None = field(default=None, repr=False)

    def to_text(self, row: dict) -> str:
        if self.format is not None:
            return self.format(row)
        return row[self.text_key]


# よく使うデータセットのプリセット。--preset で参照する。
PRESETS: dict[str, DatasetSpec] = {
    # 動作確認用（小さい順）
    "wikitext": DatasetSpec("Salesforce/wikitext", "wikitext-2-raw-v1"),  # ~4MB
    "tinystories": DatasetSpec("roneneldan/TinyStories"),  # ~2GB
    # 本番候補
    "cosmopedia": DatasetSpec("HuggingFaceTB/cosmopedia", "web_samples_v2"),
    "gsm8k": DatasetSpec(
        "openai/gsm8k",
        "main",
        format=lambda r: f"{r['question']}\n{r['answer']}",
    ),
    "wikipedia-en": DatasetSpec("wikimedia/wikipedia", "20231101.en"),
    "wikipedia-ja": DatasetSpec("wikimedia/wikipedia", "20231101.ja"),
    "fineweb-ja": DatasetSpec("HuggingFaceFW/fineweb-2", "jpn_Jpan"),
}


@dataclass
class Mixture:
    """複数ソースを文字数バジェット比で混ぜたコーパス。

    BPE は出現頻度ベースなので順番は問わない。weights は「読み出す文字数」の
    配分で、語彙がどの言語にどれだけ割かれるかを実質的に決める（バイト均衡）。
    """

    specs: list[DatasetSpec]
    weights: list[float]


# トークナイザ学習用レシピ。日本語と英語を文字数でおよそ 50:50 に均す。
RECIPES: dict[str, Mixture] = {
    "ja-en": Mixture(
        specs=[
            PRESETS["wikipedia-ja"],
            PRESETS["fineweb-ja"],
            PRESETS["wikipedia-en"],
            PRESETS["cosmopedia"],
        ],
        weights=[0.25, 0.25, 0.25, 0.25],  # JP 0.5 / EN 0.5
    ),
}


def iter_texts(
    spec: DatasetSpec,
    streaming: bool = True,
    limit: int | None = None,
    max_chars: int | None = None,
) -> Iterator[str]:
    """spec の指すデータセットから空でないテキストを1件ずつ yield する。

    max_chars を指定すると、累計文字数がそれに達した時点で打ち切る（バイト均衡用）。
    """
    ds = load_dataset(spec.path, spec.name, split=spec.split, streaming=streaming)
    if limit is not None:
        ds = ds.take(limit) if streaming else ds.select(range(min(limit, len(ds))))
    n_chars = 0
    for row in ds:
        text = spec.to_text(row)
        if text and text.strip():
            yield text
            if max_chars is not None:
                n_chars += len(text)
                if n_chars >= max_chars:
                    return


def iter_mixture(
    mix: Mixture,
    total_chars: int,
    streaming: bool = True,
) -> Iterator[str]:
    """Mixture の各ソースを weights 配分の文字数バジェットまで読んで連結する。"""
    for spec, w in zip(mix.specs, mix.weights):
        yield from iter_texts(
            spec, streaming=streaming, max_chars=int(total_chars * w)
        )


def resolve_spec(preset: str | None, dataset: str | None) -> DatasetSpec:
    """CLI 引数から DatasetSpec を解決する。

    --preset <name> か、--dataset "path[:name[:split[:text_key]]]" のいずれか。
    """
    if preset is not None:
        if preset not in PRESETS:
            raise SystemExit(
                f"unknown preset '{preset}'. choices: {', '.join(PRESETS)}"
            )
        return PRESETS[preset]
    if dataset is not None:
        path, *rest = dataset.split(":")
        name = rest[0] if len(rest) > 0 and rest[0] else None
        split = rest[1] if len(rest) > 1 and rest[1] else "train"
        text_key = rest[2] if len(rest) > 2 and rest[2] else "text"
        return DatasetSpec(path, name, split, text_key)
    raise SystemExit("either --preset or --dataset is required")
