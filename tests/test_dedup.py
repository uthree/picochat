"""Corpus hygiene (picochat.data.dedup): normalization, exact/near dedup,
benchmark decontamination, and the composed CorpusFilter -- all offline."""

import numpy as np

from picochat.data.dedup import (
    ContaminationIndex,
    CorpusFilter,
    ExactDedup,
    MinHashDedup,
    RowRepetitionFilter,
    filter_from_config,
    normalize,
    repetitive_row_mask,
    row_filter_from_config,
)


def test_normalize_folds_width_case_and_whitespace():
    # NFKC folds full-width forms -- essential for Japanese corpora
    assert normalize("Ｈｅｌｌｏ　Ｗｏｒｌｄ") == "hello world"
    assert normalize("  A\t\nB  ") == "a b"


def test_exact_dedup_catches_normalized_repeats():
    d = ExactDedup()
    assert not d.is_duplicate("Hello World")
    assert d.is_duplicate("hello   world")  # same after normalization
    assert not d.is_duplicate("hello there")


def test_minhash_catches_near_duplicates_keeps_distinct():
    d = MinHashDedup()
    base = (
        "むかしむかし、あるところにおじいさんとおばあさんが住んでいました。"
        "おじいさんは山へ柴刈りに、おばあさんは川へ洗濯に行きました。"
        "川で洗濯をしていると大きな桃が流れてきました。" * 3
    )
    assert not d.is_duplicate(base)
    # light edit: one clause appended -- still ~same shingle set
    assert d.is_duplicate(base + "おばあさんは驚きました。")
    # genuinely different document
    different = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. " * 5
    )
    assert not d.is_duplicate(different)


def test_minhash_short_docs_never_dropped():
    d = MinHashDedup(shingle=12)
    assert not d.is_duplicate("short")
    assert not d.is_duplicate("short")  # below one shingle: minhash abstains


def test_contamination_index_flags_verbatim_eval_overlap():
    idx = ContaminationIndex(n=16, stride=4)
    eval_text = (
        "アコーディオンを上手に弾くための適切なコツとして考えられないものはどれでしょう"
    )
    idx.add_eval_text(eval_text)
    # a training doc quoting the eval item verbatim (different spacing) is caught
    doc = "前置きの文章。 アコーディオンを上手に弾く ための適切なコツとして考えられないものはどれでしょう。続きの文章。"
    assert idx.is_contaminated(doc)
    assert not idx.is_contaminated(
        "全く関係のない日本語の文章です。楽器の話はしていません。"
    )


def test_contamination_stride_guarantee():
    # any shared span of >= n + stride - 1 chars must hit, wherever it starts
    idx = ContaminationIndex(n=8, stride=4)
    idx.add_eval_text("abcdefghijklmnopqrstuvwxyz")
    for offset_pad in ("", "x", "xy", "xyz"):
        assert idx.is_contaminated(offset_pad + "abcdefghijk")  # 11 = 8 + 4 - 1


def test_corpus_filter_composes_and_counts():
    f = CorpusFilter(exact=ExactDedup(), minhash=MinHashDedup())
    long_doc = "a distinct document with enough characters for shingling " * 3
    kept = f.filter_batch(["doc one text here", "doc one text here", long_doc])
    assert kept == ["doc one text here", long_doc]
    assert f.stats.docs == 3 and f.stats.exact == 1 and f.stats.dropped == 1
    assert "1" in f.stats.describe()


def test_filter_from_config():
    assert filter_from_config(None) is None
    assert filter_from_config({}) is None
    f = filter_from_config({"exact_dedup": True})
    assert f is not None and f.exact is not None and f.minhash is None
    f = filter_from_config({"minhash_dedup": {"shingle": 8}})
    assert f.minhash is not None and f.minhash.shingle == 8


# ---------------------------------------------------------------------------
# row-level repetition filter (post-packing)
# ---------------------------------------------------------------------------
def _rows(*seqs) -> np.ndarray:
    return np.array(seqs, dtype=np.uint32)


def test_repetitive_row_mask_flags_low_unique_and_long_run():
    L = 100
    rng = np.random.default_rng(0)
    diverse = rng.integers(1, 500, size=L)  # many distinct tokens -> keep
    cyclic = np.tile([1, 2, 3], L)[:L]  # 3 distinct tokens -> low unique, drop
    long_run = np.concatenate([np.full(80, 7), rng.integers(1, 500, size=L - 80)])
    rows = _rows(diverse, cyclic, long_run)
    # unique ratio 3/100 = 0.03 < 0.15 catches `cyclic`; a run of 80 > max_run
    # 50 catches `long_run`; the diverse row survives both.
    drop = repetitive_row_mask(rows, min_unique_ratio=0.15, max_run=50)
    assert drop.tolist() == [False, True, True]


def test_repetitive_row_mask_empty():
    assert repetitive_row_mask(np.empty((0, 100), dtype=np.uint32)).tolist() == []


def test_row_filter_apply_drops_and_counts():
    L = 60
    rng = np.random.default_rng(1)
    good = rng.integers(1, 400, size=L)
    constant = np.full(L, 9)  # unique ratio 1/60 and a run of 60 -> drop
    rows = _rows(good, constant, good.copy())
    rf = RowRepetitionFilter(min_unique_ratio=0.15, max_run=50)
    kept = rf.apply(rows)
    assert len(kept) == 2  # both `good` rows survive
    assert rf.rows_seen == 3 and rf.rows_dropped == 1
    assert "1" in rf.describe()


def test_row_filter_from_config():
    assert row_filter_from_config(None) is None
    assert row_filter_from_config({}) is None
    assert row_filter_from_config({"exact_dedup": True}) is None  # unrelated key
    rf = row_filter_from_config({"repetition_filter": True})
    assert rf is not None and rf.min_unique_ratio == 0.15 and rf.max_run == 512
    rf = row_filter_from_config({"repetition_filter": {"min_unique_ratio": 0.2}})
    assert rf.min_unique_ratio == 0.2
