"""Verify the reward layer end-to-end without a live judge server: the test
sandbox really runs code, the HTTP judge is exercised via a monkeypatched
POST, and composition applies the test-first gate."""

import asyncio

import pytest

from picochat import reward as R


def test_extract_code_prefers_last_fence():
    text = "explanation\n```python\nx = 1\n```\nmore\n```py\nx = 2\n```"
    assert R.extract_code(text) == "x = 2"
    assert R.extract_code("def f():\n    return 1") == "def f():\n    return 1"


def test_is_valid_python():
    assert R.is_valid_python("def f():\n    return 1")
    assert not R.is_valid_python("def f(:\n    return")


def test_run_tests_pass_and_fail():
    task = R.CodeTask(test_code="assert add(2, 3) == 5")
    good = "def add(a, b):\n    return a + b"
    bad = "def add(a, b):\n    return a - b"
    assert R.run_tests(good, task) == 1.0
    assert R.run_tests(bad, task) == 0.0


def test_run_tests_timeout_is_zero():
    task = R.CodeTask(test_code="loop()", timeout=1.0)
    code = "def loop():\n    while True:\n        pass"
    assert R.run_tests(code, task) == 0.0


def test_test_reward_extracts_then_runs():
    resp = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert (
        R.TestReward().score(resp, R.CodeTask(test_code="assert add(1, 1) == 2")) == 1.0
    )


def test_mock_judge_is_deterministic_and_bounded():
    j = R.MockJudge()
    a = asyncio.run(j.score("p", "```python\nx = 1\n```" + " " * 190))
    b = asyncio.run(j.score("p", "```python\nx = 1\n```" + " " * 190))
    assert a == b and 0.0 <= a <= 1.0
    assert asyncio.run(j.score("p", "")) == 0.0


def test_http_judge_builds_guided_request():
    # The request shape is pure (no server): guided_regex forces exactly one
    # [YN] per checklist question, and the task/response/checklist go in the
    # messages.
    j = R.HTTPJudge(questions=("q1?", "q2?", "q3?"))
    assert j._extra_body() == {"guided_regex": "[YN]{3}"}
    assert R.HTTPJudge(guided=False)._extra_body() == {}
    msgs = j._messages("add two ints", "def add(a,b): return a+b")
    assert msgs[0]["role"] == "system"
    assert "add two ints" in msgs[1]["content"]
    assert "1. q1?" in msgs[1]["content"] and "3. q3?" in msgs[1]["content"]


def test_http_judge_scores_fraction_of_yes(monkeypatch):
    # Five-question checklist; three Y -> 3/5. Guided replies are exactly the
    # letters, and stray whitespace/case doesn't matter.
    async def fake_complete(self, prompt, response):
        return "YNYNY"

    monkeypatch.setattr(R.HTTPJudge, "_complete", fake_complete)
    assert asyncio.run(R.HTTPJudge().score("task", "resp")) == 0.6


def test_http_judge_parses_unguided_yes_no_words(monkeypatch):
    # Best-effort parse of a mildly unguided reply: yes/no words start with the
    # letter we count. Two questions, one yes -> 0.5.
    async def fake_complete(self, prompt, response):
        return "1. Yes\n2. No"

    monkeypatch.setattr(R.HTTPJudge, "_complete", fake_complete)
    assert asyncio.run(R.HTTPJudge(questions=("a?", "b?")).score("t", "r")) == 0.5


def test_http_judge_survives_server_error(monkeypatch):
    async def boom(self, prompt, response):
        raise OSError("connection refused")

    monkeypatch.setattr(R.HTTPJudge, "_complete", boom)
    assert asyncio.run(R.HTTPJudge().score("t", "r")) == 0.0


def test_compose_gates_on_tests():
    # A tested task ignores the judge (test-first); pass -> w_task*1 + w_format*1.
    rm = R.RewardModel(judge=R.MockJudge())
    task = R.CodeTask(test_code="assert add(2, 2) == 4")
    resp = "```python\ndef add(a, b):\n    return a + b\n```"
    score = asyncio.run(rm.score("add two numbers", resp, task))
    assert score == pytest.approx(1.0 * 1.0 + 0.1 * 1.0)

    # No task -> falls back to the judge.
    judged = asyncio.run(rm.score("say hi", "```python\nx=1\n```", None))
    assert 0.0 <= judged <= 1.1


def test_run_tests_verbose_reports_pass_and_failure():
    task = R.CodeTask(test_code="assert add(2, 3) == 5")
    ok, out = R.run_tests_verbose("def add(a, b):\n    return a + b", task)
    assert ok is True
    bad_ok, bad_out = R.run_tests_verbose("def add(a, b):\n    return a - b", task)
    assert bad_ok is False
    assert "AssertionError" in bad_out  # failure text is captured for feedback


def test_code_agent_env_step_classifies_turns():
    env = R.CodeAgentEnv(task=R.CodeTask(test_code="assert add(1, 1) == 2"))
    good = env.step("```python\ndef add(a, b):\n    return a + b\n```")
    assert good.passed and good.valid
    wrong = env.step("```python\ndef add(a, b):\n    return a - b\n```")
    assert not wrong.passed and wrong.valid and wrong.feedback  # runs but wrong
    crash = env.step("this is not python !!!")
    assert not crash.passed and not crash.valid  # doesn't even parse


def test_trajectory_reward_prizes_eventual_success_and_stability():
    S = R.StepResult
    # crash < runs-but-wrong < pass, in per-turn quality
    assert (
        R.trajectory_reward([S(False, False)])
        < R.trajectory_reward([S(False, True)])
        < R.trajectory_reward([S(True, True)])
    )
    assert R.trajectory_reward([]) == 0.0

    # eventually reaching the answer (even after messy turns) beats never reaching it
    win_late = [S(False, False), S(False, True), S(True, True)]
    never = [S(False, True), S(False, True), S(False, True)]
    assert R.trajectory_reward(win_late) > R.trajectory_reward(never)

    # one-shot success is NOT strongly favored over eventual success (small gap),
    # and by default a longer successful episode is not punished for its length
    one_shot = [S(True, True)]
    long_win = [S(False, True)] * 5 + [S(True, True)]
    assert R.trajectory_reward(one_shot) - R.trajectory_reward(win_late) < 0.3
    assert R.trajectory_reward(long_win) >= R.trajectory_reward(win_late)


def test_trajectory_reward_step_penalty_is_opt_in():
    S = R.StepResult
    one_shot = [S(True, True)]
    long_win = [S(False, True)] * 5 + [S(True, True)]
    # default (no penalty): the long episode isn't dragged below the short one
    assert R.trajectory_reward(long_win) >= R.trajectory_reward(one_shot) - 0.3
    # opt in to a step penalty and brevity is rewarded among successes
    cfg = R.AgentRewardConfig(step_penalty=0.1)
    assert R.trajectory_reward(one_shot, cfg) > R.trajectory_reward(long_win, cfg)


def test_score_group_runs_concurrently():
    rm = R.RewardModel(judge=R.MockJudge())
    task = R.CodeTask(test_code="assert add(1, 1) == 2")
    prompts = ["p"] * 4
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    responses = [good, bad, good, bad]
    tasks = [task] * 4
    scores = asyncio.run(rm.score_group(prompts, responses, tasks))
    assert len(scores) == 4
    assert scores[0] > scores[1]  # passing rollout out-rewards failing one


def test_judge_when_tested_blends_judge_into_tested_tasks():
    class HalfJudge:
        async def score(self, prompt, response):
            return 0.5

    rm = R.RewardModel(judge=HalfJudge(), cfg=R.RewardConfig(judge_when_tested=True))
    task = R.CodeTask(test_code="assert add(2, 2) == 4")
    resp = "```python\ndef add(a, b):\n    return a + b\n```"
    score = asyncio.run(rm.score("add two numbers", resp, task))
    # base = 0.5 * 1.0 (tests pass) + 0.5 * 0.5 (judge); + w_format * 1 (valid)
    assert score == pytest.approx(1.0 * 0.75 + 0.1 * 1.0)


def test_http_judge_empty_questions_scores_zero():
    # No questions -> nothing to grade; must return 0.0 without any request
    # (this branch also guards the fraction-of-yes divide-by-zero).
    judge = R.HTTPJudge(questions=())
    assert asyncio.run(judge.score("task", "response")) == 0.0
