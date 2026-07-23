"""Generative pass@1 code eval (scripts/code_eval.py) and the shared GRPO task
loader it reuses (picochat.rl.grpo.load_tasks)."""

import torch

from picochat.inference.engine import SamplingConfig
from picochat.rl.grpo import load_tasks
from picochat.rl.reward import CodeTask
from picochat.tokenizer import BOS_TOKEN, IM_END, SPECIAL_TOKENS
from scripts.code_eval import evaluate_tasks


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
    sample gets the same scripted reply."""

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


def test_evaluate_tasks_pass_and_fail():
    # One scripted (correct) reply against a passing and a failing test: the
    # whole loop runs -- generation, code extraction, sandboxed execution --
    # and the failure record carries the test's output for inspection.
    tok = ByteTokenizer()
    reply = "```python\ndef add(a, b):\n    return a + b\n```"
    script = [*reply.encode(), tok.encode_single_token(IM_END)]
    model = ScriptedModel(script)
    samples = [
        {
            "prompt_ids": [1, 2],
            "prompt_str": "write add",
            "task": CodeTask(test_code="assert add(2, 3) == 5"),
        },
        {
            "prompt_ids": [1, 2],
            "prompt_str": "impossible",
            "task": CodeTask(test_code="assert add(2, 3) == 6"),
        },
    ]
    cfg = SamplingConfig(temperature=0.0, max_new_tokens=len(script) + 4)
    results = evaluate_tasks(model, tok, samples, cfg, "cpu", max_seq_len=4096)
    assert [r["passed"] for r in results] == [True, False]
    assert "def add" in results[0]["response"]
    assert "AssertionError" in results[1]["output"]


def test_load_tasks_reads_grpo_jsonl(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"prompt": "write add", "test": "assert add(1, 1) == 2"}\n'
        "\n"  # blank lines are skipped
        '{"prompt": "explain hash maps"}\n'
    )
    tok = ByteTokenizer()
    samples = load_tasks(str(path), tok, system="be helpful")
    assert len(samples) == 2
    assert samples[0]["prompt_str"] == "write add"
    assert samples[0]["task"].test_code == "assert add(1, 1) == 2"
    assert samples[1]["task"] is None  # no test -> judge-only task
    # rendered as a ChatML prompt: document BOS first, assistant cue last
    assert samples[0]["prompt_ids"][0] == tok.encode_single_token(BOS_TOKEN)
