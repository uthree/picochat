"""Verify the code-execution sandbox: bwrap command construction, the
require/fallback policy, and that the hardened-subprocess fallback really
enforces its rlimits and timeout. The bwrap namespace isolation itself can only
be exercised where the kernel permits namespaces, so here we assert the command
shape and drive the fallback path end-to-end."""

import subprocess
import sys
import tempfile
import warnings

import pytest

from picochat import sandbox


@pytest.fixture(autouse=True)
def _reset_sandbox_state():
    """Isolate each test from the module-global policy / one-shot warning."""
    saved = sandbox.MODE
    sandbox._warned_fallback = False
    yield
    sandbox.MODE = saved


def test_configure_rejects_unknown_mode():
    with pytest.raises(ValueError):
        sandbox.configure("jail")


def test_check_requires_working_bwrap(monkeypatch):
    monkeypatch.setattr(sandbox, "bwrap_works", lambda: False)
    sandbox.configure("bwrap")
    with pytest.raises(RuntimeError):
        sandbox.check()  # required but unavailable -> fail fast
    sandbox.configure("auto")
    sandbox.check()  # falls back, no raise
    sandbox.configure("none")
    sandbox.check()  # never uses bwrap, no raise


def test_bwrap_argv_locks_down_filesystem_and_network():
    argv = sandbox._bwrap_argv("/tmp/work-xyz", [sys.executable, "candidate.py"])
    # network + all other namespaces unshared, dies with parent, clean env
    for flag in ("--unshare-all", "--die-with-parent", "--clearenv"):
        assert flag in argv
    # the work dir is the ONLY writable bind, and it's the cwd
    i = argv.index("--bind")
    assert argv[i + 1] == "/tmp/work-xyz" and argv[i + 2] == "/tmp/work-xyz"
    j = argv.index("--chdir")
    assert argv[j + 1] == "/tmp/work-xyz"
    # the home dir and repo root are never bound (only specific system/venv paths)
    assert "/root" not in argv and "/home" not in argv
    # system dirs are mounted read-only, not writable
    assert "--ro-bind" in argv
    assert argv[-1] == "candidate.py" and argv[-2] == sys.executable


def test_fallback_runs_and_captures_output():
    sandbox.configure("none")
    with tempfile.TemporaryDirectory() as tmp:
        proc = sandbox.run(
            [sys.executable, "-c", "print('hello sandbox')"], work_dir=tmp, timeout=10
        )
    assert proc.returncode == 0
    assert "hello sandbox" in proc.stdout


def test_fallback_scrubs_environment():
    sandbox.configure("none")
    with tempfile.TemporaryDirectory() as tmp:
        proc = sandbox.run(
            [sys.executable, "-c", "import os; print(os.environ.get('SECRET','<unset>'))"],
            work_dir=tmp,
            timeout=10,
        )
    # the parent's env (incl. any secrets) is not passed through
    assert "<unset>" in proc.stdout


def test_fallback_enforces_file_size_rlimit():
    sandbox.configure("none")
    with tempfile.TemporaryDirectory() as tmp:
        proc = sandbox.run(
            [sys.executable, "-c", "open('big','w').write('x' * 10_000_000)"],
            work_dir=tmp,
            timeout=10,
            fsize_bytes=4096,
        )
    assert proc.returncode != 0  # RLIMIT_FSIZE blocks the oversized write


def test_run_raises_on_timeout():
    sandbox.configure("none")
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(subprocess.TimeoutExpired):
            sandbox.run(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                work_dir=tmp,
                timeout=1,
            )


def test_auto_mode_warns_once_on_bwrap_fallback(monkeypatch, tmp_path):
    # auto + no bubblewrap: the first run soft-falls back to the hardened
    # subprocess with a RuntimeWarning; the latch keeps later runs quiet.
    monkeypatch.setattr(sandbox, "bwrap_works", lambda: False)
    monkeypatch.setattr(sandbox, "MODE", "auto")
    monkeypatch.setattr(sandbox, "_warned_fallback", False)
    argv = [sys.executable, "-c", "print('ok')"]
    with pytest.warns(RuntimeWarning, match="bubblewrap sandbox unavailable"):
        result = sandbox.run(argv, str(tmp_path), timeout=30)
    assert result.stdout.strip() == "ok"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sandbox.run(argv, str(tmp_path), timeout=30)
    assert not [w for w in caught if w.category is RuntimeWarning]
