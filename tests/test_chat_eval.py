"""Judged chat-quality eval (scripts/chat_eval.py): the built-in prompt set,
the generate-then-judge loop (offline, with MockJudge and a scripted model),
and the aggregation math. No checkpoint, network, or GPU involved."""

import asyncio
import re

import pytest
import torch

from picochat.inference.engine import SamplingConfig
from picochat.rl.reward import MockJudge
from picochat.tokenizer import IM_END, SPECIAL_TOKENS
from scripts.chat_eval import (
    CATEGORIES,
    CHAT_QUESTIONS,
    CHAT_WEIGHTS,
    DEFAULT_PROMPTS,
    aggregate,
    evaluate_prompts,
    load_prompts,
)

# Hiragana, Katakana, and the CJK unified ideographs: any hit means the text
# is (at least partly) Japanese.
_JAPANESE = re.compile(r"[぀-ゟ゠-ヿ一-鿿]")


class ByteTokenizer:
    """1 byte = 1 token; special tokens get ids >= 256 (as in test_generate)."""

    def __init__(self):
        self._special = {tok: 256 + i for i, tok in enumerate(SPECIAL_TOKENS)}

    def encode_ordinary(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_single_token(self, token: str) -> int:
        return self._special[token]

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return bytes([token_id]) if token_id < 256 else b"?"

    def decode(self, ids: list[int]) -> str:
        joined = b"".join(self.decode_single_token_bytes(i) for i in ids)
        return joined.decode("utf-8", errors="replace")


class ScriptedModel(torch.nn.Module):
    """decode() forces a scripted token sequence under greedy sampling,
    regardless of the input ids (as in test_generate); wraps around so every
    prompt gets the same scripted reply."""

    def __init__(self, script: list[int], vocab_size: int = 300):
        super().__init__()
        self.script = list(script)
        self.vocab_size = vocab_size
        self.calls = 0

    def decode(self, x, cache=None, pos=0):
        logits = torch.full((x.shape[0], x.shape[1], self.vocab_size), -100.0)
        logits[:, -1, self.script[self.calls % len(self.script)]] = 100.0
        self.calls += 1
        return logits, None, pos + x.shape[1]


# ---------------------------------------------------------------------------
# the built-in prompt set
# ---------------------------------------------------------------------------
def test_builtin_prompts_cover_all_categories_bilingually():
    prompts = load_prompts(DEFAULT_PROMPTS)
    assert len(prompts) >= 40
    assert all(p["prompt"].strip() for p in prompts)
    # every MT-Bench category is present, and nothing outside the set
    assert {p["category"] for p in prompts} == set(CATEGORIES)
    # bilingual: both Japanese and English prompts exist, in real proportion
    # (not one token gesture at each language)
    japanese = sum(bool(_JAPANESE.search(p["prompt"])) for p in prompts)
    assert 10 <= japanese <= len(prompts) - 10


def test_load_prompts_limit_and_validation(tmp_path):
    assert len(load_prompts(DEFAULT_PROMPTS, limit=3)) == 3
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"category": "sorcery", "prompt": "cast a spell"}\n')
    with pytest.raises(ValueError, match="sorcery"):
        load_prompts(str(bad))
    missing = tmp_path / "missing.jsonl"
    missing.write_text('{"category": "math"}\n')
    with pytest.raises(ValueError, match="prompt"):
        load_prompts(str(missing))


def test_chat_checklist_shape():
    # HTTPJudge raises at score time on a weights/questions length mismatch;
    # catch a drive-by edit here instead, and keep every weight positive so
    # the weighted fraction stays in [0, 1].
    assert len(CHAT_QUESTIONS) == len(CHAT_WEIGHTS)
    assert all(w > 0 for w in CHAT_WEIGHTS)


# ---------------------------------------------------------------------------
# evaluate_prompts: generate -> judge, end to end (offline)
# ---------------------------------------------------------------------------
def _scripted_setup(reply: str):
    tok = ByteTokenizer()
    script = [*reply.encode(), tok.encode_single_token(IM_END)]
    model = ScriptedModel(script)
    sampling = SamplingConfig(temperature=0.0, max_new_tokens=len(script) + 4)
    return tok, model, sampling


def test_evaluate_prompts_end_to_end_with_mock_judge():
    reply = "```python\nprint('hi')\n```"
    tok, model, sampling = _scripted_setup(reply)
    prompts = [
        {"category": "coding", "prompt": "print something"},
        {"category": "writing", "prompt": "俳句を詠んでください"},
    ]
    results = evaluate_prompts(
        model,
        tok,
        prompts,
        sampling,
        MockJudge(),
        "cpu",
        max_seq_len=4096,
        concurrency=2,
    )
    assert [r["category"] for r in results] == ["coding", "writing"]
    assert all(r["prompt"] == p["prompt"] for r, p in zip(results, prompts))
    # the scripted reply came through generation + decode intact...
    assert all(r["response"] == reply for r in results)
    # ...and MockJudge scored it: non-empty valid python earns a real score
    assert all(0.0 < r["score"] <= 1.0 for r in results)


def test_evaluate_prompts_respects_concurrency_bound():
    # A judge that tracks its own in-flight count: the semaphore must keep it
    # at or under the requested bound even though all pairs are gathered.
    class CountingJudge:
        def __init__(self):
            self.in_flight = 0
            self.peak = 0
            self.pairs = []

        async def score(self, prompt: str, response: str) -> float:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            self.pairs.append((prompt, response))
            await asyncio.sleep(0)  # yield so other tasks could overlap
            self.in_flight -= 1
            return 0.5

    tok, model, sampling = _scripted_setup("ok")
    prompts = [{"category": "stem", "prompt": f"q{i}"} for i in range(6)]
    judge = CountingJudge()
    results = evaluate_prompts(
        model, tok, prompts, sampling, judge, "cpu", max_seq_len=4096, concurrency=2
    )
    assert judge.peak <= 2
    # every (prompt, reply) pair reached the judge, and scores landed
    assert sorted(p for p, _ in judge.pairs) == sorted(p["prompt"] for p in prompts)
    assert [r["score"] for r in results] == [0.5] * 6


# ---------------------------------------------------------------------------
# aggregation math
# ---------------------------------------------------------------------------
def test_aggregate_per_category_and_overall_means():
    results = [
        {"category": "math", "score": 1.0},
        {"category": "math", "score": 0.5},
        {"category": "writing", "score": 0.0},
    ]
    summary = aggregate(results)
    assert summary["categories"] == {"math": 0.75, "writing": 0.0}
    # overall is the mean over all results, NOT the mean of category means
    assert summary["overall"] == pytest.approx(0.5)
