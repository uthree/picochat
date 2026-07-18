"""Reward functions for GRPO-style RL post-training (consumed by the GRPO trainer).

Two reward sources, composed with a test-first gate by `RewardModel`:

- `TestReward`  -- run the response's code against unit tests in a subprocess
  sandbox and return whether it passed. Verifiable, deterministic, no external
  calls; this is the backbone of the reward.
- `HTTPJudge` / `MockJudge` -- score the response with an external open-weight
  LLM served over an OpenAI-compatible endpoint (e.g. `vllm serve
  Qwen/Qwen2.5-7B-Instruct --port 8001`), or a deterministic stand-in for
  single-GPU verification. Used only where tests can't reach (a task ships no
  tests, or we want to grade style / natural-language correctness).

Everything here is decoupled from the policy model AND from the serving
backend: the judge speaks the OpenAI chat-completions API, so the same code
runs against a `MockJudge` (single-GPU verification) or a large vLLM pod (H100
production) by changing only the base URL / model name in config.

The reward layer never raises into the training loop: a sandbox timeout, a
crashed judge server or an unparseable score all degrade to 0.0 so a bad
rollout simply earns no reward instead of killing the run.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from openai import AsyncOpenAI

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the last fenced code block out of a model response, or return the
    whole text if it has no fences (the model may answer with bare code)."""
    blocks = _CODE_FENCE.findall(text)
    return blocks[-1].strip() if blocks else text.strip()


def is_valid_python(code: str) -> bool:
    """Cheap, call-free validity signal: does the code even parse?  Used as a
    small format-shaping reward term so a syntactically broken rollout ranks
    below a valid-but-wrong one."""
    try:
        compile(code, "<response>", "exec")
        return True
    except SyntaxError:
        return False


@dataclass
class CodeTask:
    """The verifiable part of a prompt: `test_code` is appended after the
    response's code and the whole file is executed; a clean exit (assertions
    hold, no exception) counts as a pass.  `setup` is prepended (imports /
    fixtures) if a task needs it."""

    test_code: str
    setup: str = ""
    timeout: float = 10.0


def run_tests(code: str, task: CodeTask) -> float:
    """Execute `setup + code + test_code` in a subprocess and return 1.0 on a
    clean exit, else 0.0.  Runs in a throwaway temp dir with a wall-clock
    timeout; any failure mode (assertion, exception, timeout, unparseable
    code) is a 0.0 rather than an exception into the caller.

    Binary pass/fail is deliberate for a first cut; swap the runner for pytest
    and count passed/total here if you want a graded pass-rate.
    """
    script = f"{task.setup}\n{code}\n{task.test_code}\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(script)
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                timeout=task.timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return 0.0
        return 1.0 if proc.returncode == 0 else 0.0


@dataclass
class TestReward:
    """Verifiable code reward: extract code from the response, run the task's
    tests, return pass (1.0) / fail (0.0)."""

    def score(self, response: str, task: CodeTask) -> float:
        return run_tests(extract_code(response), task)


# --- External LLM judge (OpenAI-compatible endpoint) ---------------------------


class JudgeBackend(Protocol):
    """Anything that can turn (prompt, response) into a scalar in [0, 1]."""

    async def score(self, prompt: str, response: str) -> float: ...


_DEFAULT_RUBRIC = (
    "You are a strict grader. You are given a task, a response, and a numbered "
    "checklist of yes/no questions. Judge the response against each question in "
    "order and answer with a single letter -- Y for yes, N for no. Reply with "
    "ONLY those letters, one per question, no spaces, punctuation, or other text "
    "(e.g. 'YNYY')."
)

# A checklist beats a single 0-10 score: each item is a concrete, near-binary
# judgement, and summing the yeses is far less noisy than asking one model to
# pick a calibrated integer. These defaults are general (the judge only grades
# prompts the tests can't reach); override `questions` in config per task family.
_DEFAULT_QUESTIONS = (
    "Does the response directly address what the task asks for?",
    "Is the response's answer or solution correct?",
    "Is the response complete, without leaving the task half-done?",
    "If the response includes code, is it valid and runnable (answer Y if it "
    "includes no code)?",
    "Is the response clear, well-structured, and free of irrelevant filler?",
)


@dataclass
class HTTPJudge:
    """Score responses via an external open-weight model on an OpenAI-compatible
    endpoint (vLLM/SGLang/Ollama/hosted -- all the same wire format).

    Grading is a yes/no checklist: the judge answers each of `questions` with Y
    or N, and the score is the fraction answered Y (already in [0, 1]). When the
    server supports guided decoding (vLLM), `guided` constrains the reply to
    exactly one [YN] letter per question via a regex, so parsing can't fail;
    without it, leading Y/N letters are counted best-effort. temperature=0 keeps
    rewards stable across identical rollouts. The `AsyncOpenAI` client is created
    lazily so constructing an HTTPJudge (e.g. from config) never opens a
    connection.
    """

    base_url: str = "http://localhost:8001/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key: str = "dummy"  # vLLM ignores the value
    rubric: str = _DEFAULT_RUBRIC
    questions: tuple[str, ...] = _DEFAULT_QUESTIONS
    guided: bool = True  # send vLLM's guided_regex; harmless if unsupported
    timeout: float = 30.0
    _client: AsyncOpenAI | None = field(default=None, init=False, repr=False)

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
            )
        return self._client

    def _messages(self, prompt: str, response: str) -> list[dict]:
        checklist = "\n".join(f"{i}. {q}" for i, q in enumerate(self.questions, 1))
        return [
            {"role": "system", "content": self.rubric},
            {
                "role": "user",
                "content": (
                    f"# Task\n{prompt}\n\n# Response\n{response}\n\n"
                    f"# Checklist\n{checklist}"
                ),
            },
        ]

    def _extra_body(self) -> dict:
        # vLLM reads guided_regex to force exactly one [YN] per question; other
        # servers ignore the unknown key.
        if not self.guided:
            return {}
        return {"guided_regex": f"[YN]{{{len(self.questions)}}}"}

    async def _complete(self, prompt: str, response: str) -> str:
        r = await self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, response),
            temperature=0,
            max_tokens=len(self.questions) + 4,
            extra_body=self._extra_body(),
        )
        return r.choices[0].message.content or ""

    async def score(self, prompt: str, response: str) -> float:
        n = len(self.questions)
        if n == 0:
            return 0.0
        try:
            text = await self._complete(prompt, response)
        except Exception:  # server down / API error -> no reward, never crash
            return 0.0
        # Count leading Y/N letters (exact when guided; the words yes/no also
        # start with the right letter for a mildly unguided reply). Missing
        # answers count as N: score is (# yes) / (# questions).
        letters = re.findall(r"[YN]", text.upper())[:n]
        return sum(c == "Y" for c in letters) / n


@dataclass
class MockJudge:
    """Deterministic stand-in for a real judge, for wiring up and verifying the
    GRPO loop on a single GPU where a full judge model won't co-fit.  Rewards
    valid, non-trivial, appropriately-sized responses -- enough signal to prove
    reward -> advantage -> loss flows, not a real quality measure."""

    target_len: int = 200

    async def score(self, prompt: str, response: str) -> float:
        if not response.strip():
            return 0.0
        valid = 1.0 if is_valid_python(extract_code(response)) else 0.3
        # triangular preference around target_len, in [0, 1]
        closeness = max(
            0.0, 1.0 - abs(len(response) - self.target_len) / self.target_len
        )
        return round(0.5 * valid + 0.5 * closeness, 4)


# --- Composition ---------------------------------------------------------------


@dataclass
class RewardConfig:
    """Weights for the test-first composition (see `compose`)."""

    w_task: float = 1.0  # the gated task reward (tests if present, else judge)
    w_format: float = 0.1  # cheap validity shaping, always applied
    judge_when_tested: bool = False  # also fold judge into tasks that have tests


@dataclass
class RewardModel:
    """Compose the test backbone with the external judge under a test-first
    gate: a task with tests is scored by the tests (no external call); a task
    without tests falls back to the judge.  A small validity term shapes both.
    Returns one scalar per response, which GRPO then normalizes within its
    rollout group."""

    judge: JudgeBackend
    test: TestReward = field(default_factory=TestReward)
    cfg: RewardConfig = field(default_factory=RewardConfig)

    async def score(self, prompt: str, response: str, task: CodeTask | None) -> float:
        fmt = 1.0 if is_valid_python(extract_code(response)) else 0.0
        if task is not None and task.test_code:
            # run_tests spawns a subprocess; keep it off the event loop so a
            # rollout group's tests and judge calls overlap (see score_group).
            base = await asyncio.to_thread(self.test.score, response, task)
            if self.cfg.judge_when_tested:
                base = 0.5 * base + 0.5 * await self.judge.score(prompt, response)
        else:
            base = await self.judge.score(prompt, response)
        return self.cfg.w_task * base + self.cfg.w_format * fmt

    async def score_group(
        self,
        prompts: list[str],
        responses: list[str],
        tasks: list[CodeTask | None],
        concurrency: int = 32,
    ) -> list[float]:
        """Score a whole GRPO rollout group concurrently (bounded), so the
        judge calls for one prompt's N samples overlap instead of serializing.
        Tests run in threads too (subprocess), so this parallelizes both paths.
        """
        sem = asyncio.Semaphore(concurrency)

        async def one(p: str, r: str, t: CodeTask | None) -> float:
            async with sem:
                return await self.score(p, r, t)

        return await asyncio.gather(
            *(one(p, r, t) for p, r, t in zip(prompts, responses, tasks))
        )
