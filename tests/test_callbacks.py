"""BenchmarkEvalCallback: cadence, rank gating, logging, error resilience --
with evaluate_task stubbed out (no datasets, no network)."""

import pytest
import torch

from picochat.training import callbacks as C


class StubTrainer:
    def __init__(self, step=0, zero=True):
        self.global_step = step
        self.is_global_zero = zero


class StubModule:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.device = "cpu"
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = value


@pytest.fixture()
def fake_eval(monkeypatch):
    calls = []

    def fake(model, tokenizer, task, **kwargs):
        calls.append((task, kwargs))
        return {"task": task, "n": 10, "acc": 0.5, "acc_norm": 0.6, "random": 0.25}

    monkeypatch.setattr(C, "evaluate_task", fake)
    return calls


def _cb(**kw):
    defaults = dict(tokenizer=object(), tasks=["hellaswag"], every_n_steps=10)
    return C.BenchmarkEvalCallback(**{**defaults, **kw})


def test_callback_runs_on_schedule_and_logs(fake_eval):
    cb = _cb(tasks=["hellaswag", "arc_easy"], limit=50, chat=True)
    module = StubModule()
    cb.on_train_batch_end(StubTrainer(step=10), module, None, None, 0)
    assert [t for t, _ in fake_eval] == ["hellaswag", "arc_easy"]
    assert fake_eval[0][1]["limit"] == 50 and fake_eval[0][1]["chat"] is True
    assert module.logged["bench/hellaswag/acc"] == 0.5
    assert module.logged["bench/arc_easy/acc_norm"] == 0.6


def test_callback_cadence(fake_eval):
    cb = _cb(every_n_steps=10)
    module = StubModule()
    for step in (0, 3, 9):  # step 0 and non-multiples: no eval
        cb.on_train_batch_end(StubTrainer(step=step), module, None, None, 0)
    assert fake_eval == []
    cb.on_train_batch_end(StubTrainer(step=10), module, None, None, 0)
    # the same global_step repeats across accumulate microbatches: run once
    cb.on_train_batch_end(StubTrainer(step=10), module, None, None, 1)
    assert len(fake_eval) == 1
    cb.on_train_batch_end(StubTrainer(step=20), module, None, None, 0)
    assert len(fake_eval) == 2


def test_callback_rank_zero_only(fake_eval):
    cb = _cb()
    cb.on_train_batch_end(StubTrainer(step=10, zero=False), StubModule(), None, None, 0)
    assert fake_eval == []


def test_callback_restores_train_mode_and_survives_errors(monkeypatch):
    def boom(*a, **k):
        raise OSError("hub down")

    monkeypatch.setattr(C, "evaluate_task", boom)
    cb = _cb()
    module = StubModule()
    module.model.train()
    cb.on_train_batch_end(StubTrainer(step=10), module, None, None, 0)  # no raise
    assert module.model.training  # flipped back to train mode
    assert module.logged == {}


def test_benchmark_callback_from_config():
    assert C.benchmark_callback_from_config({}, object(), chat=False) is None
    cb = C.benchmark_callback_from_config(
        {"benchmark_eval": {"tasks": ["boolq"], "every_n_steps": 7, "limit": None}},
        object(),
        chat=True,
    )
    assert cb.tasks == ["boolq"] and cb.every_n_steps == 7
    assert cb.limit is None and cb.chat is True


def test_nan_guard_stops_on_nonfinite():
    import torch

    from picochat.training.callbacks import NaNGuardCallback

    class T:
        def __init__(self):
            self.global_step = 10
            self.should_stop = False

    cb = NaNGuardCallback()
    tr = T()
    # finite loss: keeps going
    cb.on_train_batch_end(tr, None, torch.tensor(2.5), None, 0)
    assert tr.should_stop is False
    # dict-wrapped finite loss too
    cb.on_train_batch_end(tr, None, {"loss": torch.tensor(1.0)}, None, 0)
    assert tr.should_stop is False
    # NaN: stops immediately (patience 1)
    cb.on_train_batch_end(tr, None, torch.tensor(float("nan")), None, 0)
    assert tr.should_stop is True


def test_nan_guard_patience():
    import torch

    from picochat.training.callbacks import NaNGuardCallback

    class T:
        global_step = 5
        should_stop = False

    cb = NaNGuardCallback(patience=2)
    tr = T()
    cb.on_train_batch_end(tr, None, torch.tensor(float("inf")), None, 0)
    assert tr.should_stop is False  # 1 bad, tolerated
    cb.on_train_batch_end(tr, None, torch.tensor(float("nan")), None, 0)
    assert tr.should_stop is True  # 2nd consecutive bad -> stop
    # a good step resets the counter
    cb2 = NaNGuardCallback(patience=2)
    tr2 = T()
    cb2.on_train_batch_end(tr2, None, torch.tensor(float("nan")), None, 0)
    cb2.on_train_batch_end(tr2, None, torch.tensor(1.0), None, 0)  # reset
    cb2.on_train_batch_end(tr2, None, torch.tensor(float("nan")), None, 0)
    assert tr2.should_stop is False  # only 1 bad since reset
