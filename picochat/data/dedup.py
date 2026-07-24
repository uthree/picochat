"""Corpus hygiene for pretraining data prep: exact dedup, MinHash near-dedup,
and benchmark decontamination, applied as a streaming text filter inside
scripts/base_setup.py (config `filter:` section).

Everything is character-shingle based rather than word based, deliberately:
the corpus is CJK-heavy and Japanese/Chinese have no word boundaries, so word
n-grams would silently degrade to whole-sentence tokens there. Character
shingles behave uniformly across scripts.

- ExactDedup      -- 64-bit hash of the normalized text; drops byte-identical
                     (post-normalization) repeats. Cheap and exact.
- MinHashDedup    -- MinHash-LSH over character shingles; drops *near*
                     duplicates (boilerplate re-crawls, light edits). In
                     memory: fine for tens of millions of documents; the
                     signature/band tables are the only state.
- ContaminationIndex -- normalized character n-grams of the benchmark items
                     (picochat.evals.tasks); any document containing a long
                     enough verbatim overlap with an eval item is dropped, so
                     benchmark numbers stay measurements rather than recall.
- CorpusFilter    -- bundles the three with kept/dropped counters; one
                     instance spans a whole base_setup recipe run, so dedup
                     is corpus-wide (across datasets), not per dataset.

All state is in-process; base_setup is a single-process pipeline.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

_WS = re.compile(r"\s+")

# 2^61 - 1, a Mersenne prime: the universal-hash modulus for the MinHash
# permutations (a*x + b mod p over 64-bit shingle hashes).
_MERSENNE = (1 << 61) - 1


def normalize(text: str) -> str:
    """Canonical form for hashing: NFKC (full-width/half-width and
    compatibility forms collapse -- essential for Japanese text), casefold,
    whitespace runs collapsed to one space."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _hash64(data: str) -> int:
    return int.from_bytes(hashlib.blake2b(data.encode(), digest_size=8).digest(), "big")


class ExactDedup:
    """Drop documents whose normalized text was already seen."""

    def __init__(self) -> None:
        self._seen: set[int] = set()

    def is_duplicate(self, text: str) -> bool:
        h = _hash64(normalize(text))
        if h in self._seen:
            return True
        self._seen.add(h)
        return False


class MinHashDedup:
    """MinHash-LSH near-duplicate detection over character shingles.

    With `bands` bands of `rows` rows (bands*rows permutations) the LSH match
    probability crosses 1/2 near a Jaccard similarity of (1/bands)^(1/rows)
    -- the defaults (8x8) target ~0.77, i.e. documents sharing roughly three
    quarters of their shingles collide. A candidate colliding in ANY band is
    treated as a duplicate (no verification pass: at these thresholds false
    positives are rare and this is lossy corpus pruning, not retrieval).
    Documents shorter than one shingle are never dropped here."""

    def __init__(self, shingle: int = 12, bands: int = 8, rows: int = 8, seed: int = 0):
        self.shingle = shingle
        self.bands = bands
        self.rows = rows
        n = bands * rows
        # Deterministic permutation parameters derived from the seed.
        self._params = [
            (
                _hash64(f"minhash-a-{seed}-{i}") % _MERSENNE or 1,
                _hash64(f"minhash-b-{seed}-{i}") % _MERSENNE,
            )
            for i in range(n)
        ]
        self._tables: list[dict[tuple[int, ...], None]] = [{} for _ in range(bands)]

    def _signature(self, text: str) -> list[int] | None:
        norm = normalize(text)
        if len(norm) < self.shingle:
            return None
        shingles = {
            _hash64(norm[i : i + self.shingle])
            for i in range(len(norm) - self.shingle + 1)
        }
        return [min((a * s + b) % _MERSENNE for s in shingles) for a, b in self._params]

    def is_duplicate(self, text: str) -> bool:
        sig = self._signature(text)
        if sig is None:
            return False
        keys = [
            tuple(sig[band * self.rows : (band + 1) * self.rows])
            for band in range(self.bands)
        ]
        duplicate = any(key in self._tables[band] for band, key in enumerate(keys))
        for band, key in enumerate(keys):
            self._tables[band][key] = None
        return duplicate


class ContaminationIndex:
    """Verbatim-overlap decontamination against benchmark items.

    Eval texts are normalized with whitespace REMOVED entirely (so different
    wrapping/spacing can't hide an overlap; also the natural form for
    unspaced Japanese), then every character n-gram (stride 1) is indexed.
    Documents are probed at `stride` positions: any shared verbatim span of
    at least n + stride - 1 characters is guaranteed to hit, at 1/stride the
    probing cost. n=32 (~5-8 English words, ~16 CJK characters) with
    stride 8 flags spans of ~39+ characters -- long enough to be a real leak
    rather than a common phrase."""

    def __init__(self, n: int = 32, stride: int = 8):
        self.n = n
        self.stride = stride
        self._grams: set[int] = set()

    @staticmethod
    def _squash(text: str) -> str:
        return _WS.sub("", unicodedata.normalize("NFKC", text).casefold())

    def add_eval_text(self, text: str) -> None:
        s = self._squash(text)
        for i in range(len(s) - self.n + 1):
            self._grams.add(_hash64(s[i : i + self.n]))

    @classmethod
    def from_eval_tasks(
        cls,
        tasks: list[str] | None = None,
        n: int = 32,
        stride: int = 8,
        limit: int | None = None,
    ) -> "ContaminationIndex":
        """Index every (context, completion) text of the given benchmark
        tasks (default: all registered tasks). Downloads the eval sets on
        first use -- base_setup already assumes Hub access."""
        from picochat.evals.tasks import TASKS, load_task_examples

        index = cls(n=n, stride=stride)
        for task in tasks if tasks is not None else list(TASKS):
            for ex in load_task_examples(task, limit=limit):
                for ctx, completion in ex.choices:
                    index.add_eval_text(ctx + completion)
        return index

    def is_contaminated(self, text: str) -> bool:
        if not self._grams:
            return False
        s = self._squash(text)
        if len(s) < self.n:
            return False
        return any(
            _hash64(s[i : i + self.n]) in self._grams
            for i in range(0, len(s) - self.n + 1, self.stride)
        )


@dataclass
class FilterStats:
    docs: int = 0
    exact: int = 0
    near: int = 0
    contaminated: int = 0

    @property
    def dropped(self) -> int:
        return self.exact + self.near + self.contaminated

    def describe(self) -> str:
        if self.docs == 0:
            return "filter: no documents seen"
        return (
            f"filter: dropped {self.dropped:,}/{self.docs:,} docs "
            f"({self.dropped / self.docs:.2%}) -- "
            f"{self.exact:,} exact dup, {self.near:,} near dup, "
            f"{self.contaminated:,} eval-contaminated"
        )


@dataclass
class CorpusFilter:
    """The composed streaming filter base_setup applies per text batch.
    Order: exact (cheapest) -> near-dup -> contamination; a document dropped
    by an earlier stage never reaches (or pollutes the state of) a later
    one. One instance per recipe run, so deduplication spans datasets."""

    exact: ExactDedup | None = None
    minhash: MinHashDedup | None = None
    contamination: ContaminationIndex | None = None
    stats: FilterStats = field(default_factory=FilterStats)

    def keep(self, text: str) -> bool:
        self.stats.docs += 1
        if self.exact is not None and self.exact.is_duplicate(text):
            self.stats.exact += 1
            return False
        if self.minhash is not None and self.minhash.is_duplicate(text):
            self.stats.near += 1
            return False
        if self.contamination is not None and self.contamination.is_contaminated(text):
            self.stats.contaminated += 1
            return False
        return True

    def filter_batch(self, texts: list[str]) -> list[str]:
        return [t for t in texts if self.keep(t)]


def filter_from_config(cfg: dict | None) -> CorpusFilter | None:
    """Build the CorpusFilter a base_setup recipe's `filter:` section asks
    for (None -> no filtering, the previous behavior):

        filter:
            exact_dedup: true
            minhash_dedup: true       # optional {shingle, bands, rows}
            decontaminate: true       # or a list of task names
    """
    if not cfg:
        return None
    exact = ExactDedup() if cfg.get("exact_dedup") else None
    minhash = None
    if mh := cfg.get("minhash_dedup"):
        kwargs = mh if isinstance(mh, dict) else {}
        minhash = MinHashDedup(**kwargs)
    contamination = None
    if decon := cfg.get("decontaminate"):
        tasks = decon if isinstance(decon, list) else None
        contamination = ContaminationIndex.from_eval_tasks(tasks)
    if exact is None and minhash is None and contamination is None:
        return None
    return CorpusFilter(exact=exact, minhash=minhash, contamination=contamination)


# ---------------------------------------------------------------------------
# Row-level repetition filter (post-packing)
# ---------------------------------------------------------------------------
# The dedup above works on document *text*. This one works on the packed token
# *rows* (block_size+1 wide) that actually feed the model, and drops the
# degenerate ones -- rows dominated by a tiny set of tokens or by a long run of
# a single token. Such rows arise from repetitive/boilerplate source (heavily
# so in raw code corpora like the-stack: measured ~69% of its packed rows) and
# push the linear-attention (Gated DeltaNet) bf16 backward into overflow, i.e.
# a non-finite gradient. The trainer's skip-on-non-finite guard already keeps
# those from poisoning the run, but every skipped step is wasted compute;
# removing the rows up front recovers it. Both metrics are fully vectorized
# numpy over the whole (n_rows, row_len) batch -- microseconds per row, versus
# the Python-serial MinHash path -- so this is cheap enough to always leave on.


def repetitive_row_mask(
    rows: np.ndarray,
    min_unique_ratio: float = 0.15,
    max_run: int = 512,
) -> np.ndarray:
    """Boolean mask (True = DROP) over `rows` (shape (n, row_len)) flagging the
    degenerate rows: those whose distinct-token ratio falls below
    `min_unique_ratio` (cyclic / tiny-vocabulary repetition), or that contain a
    run of one identical token longer than `max_run` (which is what actually
    blows up the Gated DeltaNet recurrent state). Empty input -> empty mask."""
    n = len(rows)
    if n == 0:
        return np.zeros(0, dtype=bool)
    row_len = rows.shape[1]

    # distinct-token ratio, without a Python per-row np.unique: sort each row
    # and count the positions where the sorted value changes.
    sorted_rows = np.sort(rows, axis=1)
    n_unique = (np.diff(sorted_rows, axis=1) != 0).sum(axis=1) + 1  # (n,)
    low_unique = (n_unique / row_len) < min_unique_ratio

    # longest run of an identical token per row, vectorized: for each position
    # track the index of the last value-change; the gap to it is the current
    # run length, and the per-row max is the longest run.
    idx = np.arange(row_len)[None, :]
    change = np.ones_like(rows, dtype=bool)
    change[:, 1:] = rows[:, 1:] != rows[:, :-1]
    last_change = np.maximum.accumulate(np.where(change, idx, 0), axis=1)
    longest_run = (idx - last_change + 1).max(axis=1)  # (n,)
    long_run = longest_run > max_run

    return low_unique | long_run


@dataclass
class RowRepetitionFilter:
    """Streaming row filter applied by base_setup after packing, before a batch
    of rows is written to the shard. Carries the same thresholds as
    `repetitive_row_mask` plus running counts for the end-of-run summary. One
    instance per recipe run."""

    min_unique_ratio: float = 0.15
    max_run: int = 512
    rows_seen: int = 0
    rows_dropped: int = 0

    def apply(self, rows: np.ndarray) -> np.ndarray:
        """Return `rows` with the degenerate ones removed (a view/copy of the
        kept rows), updating the counts."""
        self.rows_seen += len(rows)
        drop = repetitive_row_mask(rows, self.min_unique_ratio, self.max_run)
        self.rows_dropped += int(drop.sum())
        return rows[~drop]

    def describe(self) -> str:
        if self.rows_seen == 0:
            return "row filter: no rows seen"
        return (
            f"row filter: dropped {self.rows_dropped:,}/{self.rows_seen:,} "
            f"packed rows ({self.rows_dropped / self.rows_seen:.2%}) as "
            f"degenerate (unique<{self.min_unique_ratio}, run>{self.max_run})"
        )


def row_filter_from_config(cfg: dict | None) -> RowRepetitionFilter | None:
    """Build the post-packing row filter from a base_setup recipe's `filter:`
    section (None -> off):

        filter:
            repetition_filter: true                 # defaults, or a dict:
            # repetition_filter: {min_unique_ratio: 0.15, max_run: 512}

    Independent of the text-level dedup keys in the same section; either can be
    used without the other."""
    if not cfg:
        return None
    rf = cfg.get("repetition_filter")
    if not rf:
        return None
    kwargs = rf if isinstance(rf, dict) else {}
    return RowRepetitionFilter(**kwargs)
