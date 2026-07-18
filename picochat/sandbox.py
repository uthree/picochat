"""Isolated execution of untrusted, model-generated code for the RL reward.

GRPO runs code the *policy* invented; `picochat.reward.run_tests` would otherwise
execute it with the trainer's own user, filesystem, and network. This wraps
execution in a [bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`)
sandbox: a fresh mount + network + pid namespace with only read-only system
directories and the Python runtime visible, a writable bind for the throwaway
work dir, a tmpfs `/tmp`, **no network** (unshared), and **no view of the home
dir, the repo, or the checkpoints**. On top of that every run gets POSIX
rlimits (address space, CPU seconds, file size, process count) and a scrubbed
environment.

`bwrap` needs working user namespaces (unprivileged, or root with CAP_SYS_ADMIN).
Where the kernel forbids them -- some nested containers block the `unshare`
syscall outright -- we fall back to a plain subprocess still hardened with the
rlimits and scrubbed env (but *without* the filesystem/network isolation) and
warn once. Set `PICOCHAT_SANDBOX=bwrap` (or `sandbox: bwrap` in the GRPO config)
to make a missing sandbox a hard error instead of a soft fallback; `none`
disables bwrap and uses the hardened subprocess directly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import warnings
from functools import lru_cache

# Resolved isolation policy: "auto" (bwrap if it works, else fall back),
# "bwrap" (require it) or "none" (hardened subprocess only). Env var seeds the
# default; configure() lets the trainer override it from a config file.
MODE: str = os.environ.get("PICOCHAT_SANDBOX", "auto")

# Defaults for the per-run resource limits (overridable per call).
DEFAULT_MEM_BYTES = 2 * 1024**3  # address space ceiling (RLIMIT_AS)
DEFAULT_FSIZE_BYTES = 64 * 1024**2  # largest file the code may write
DEFAULT_NPROC = 64  # process/thread count (fork-bomb guard)

_warned_fallback = False


def configure(mode: str) -> None:
    """Set the isolation policy ('auto' | 'bwrap' | 'none')."""
    global MODE
    if mode not in ("auto", "bwrap", "none"):
        raise ValueError(f"unknown sandbox mode {mode!r} (auto|bwrap|none)")
    MODE = mode


@lru_cache(maxsize=1)
def bwrap_works() -> bool:
    """True if `bwrap` exists AND can actually create namespaces here (cached).
    Probes with a trivial sandboxed `true`, since the binary being installed
    doesn't mean the kernel/container permits unprivileged namespaces."""
    if shutil.which("bwrap") is None:
        return False
    try:
        proc = subprocess.run(
            [
                "bwrap",
                "--ro-bind", "/usr", "/usr",
                "--ro-bind", "/bin", "/bin",
                "--unshare-all",
                "--die-with-parent",
                "/bin/true",
            ],
            capture_output=True,
            timeout=10,
        )
        return proc.returncode == 0
    except Exception:
        return False


def check() -> None:
    """Fail fast at startup if the resolved policy requires bwrap but it can't
    run. Call once before training so a missing sandbox surfaces immediately
    instead of silently zeroing every reward."""
    if MODE == "bwrap" and not bwrap_works():
        raise RuntimeError(
            "sandbox mode 'bwrap' requested but bubblewrap cannot create "
            "namespaces here (install bubblewrap and enable unprivileged user "
            "namespaces, or run where CAP_SYS_ADMIN is available). Set "
            "PICOCHAT_SANDBOX=auto to fall back to a hardened subprocess."
        )


def _system_ro_binds() -> list[str]:
    """Read-only binds for the system dirs and the Python runtime (interpreter,
    stdlib, and the active venv), so the sandbox can run Python but sees nothing
    of the home dir / repo / weights beyond these paths."""
    paths = ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"]
    # The interpreter may live in a venv outside /usr (e.g. .venv under the
    # repo); bind just its prefixes, not the whole repo.
    paths.append(sys.base_prefix)
    paths.append(sys.prefix)
    paths.append(os.path.dirname(os.path.realpath(sys.executable)))
    seen: set[str] = set()
    binds: list[str] = []
    for p in paths:
        rp = os.path.realpath(p)
        if rp in seen or not os.path.exists(rp):
            continue
        seen.add(rp)
        binds += ["--ro-bind", rp, rp]
    return binds


def _bwrap_argv(work_dir: str, argv: list[str]) -> list[str]:
    """Build the bwrap command wrapping `argv`, with `work_dir` as the only
    writable path (bound at its real path so absolute paths in argv resolve)."""
    return [
        "bwrap",
        "--unshare-all",  # net, pid, ipc, uts, cgroup, user namespaces
        "--die-with-parent",
        "--new-session",  # own session: no terminal escape via TIOCSTI
        *_system_ro_binds(),
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--bind", work_dir, work_dir,
        "--chdir", work_dir,
        "--clearenv",
        "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "HOME", work_dir,
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "LANG", "C.UTF-8",
        "--",
        *argv,
    ]


def _rlimit_preexec(mem_bytes: int, cpu_seconds: int, fsize_bytes: int, nproc: int):
    """preexec_fn that caps the child's memory, CPU time, file size and process
    count. Inherited across bwrap's exec, so it bounds the sandboxed code too."""
    import resource

    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
        try:  # NPROC can be unsettable under some namespace configs
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        except (ValueError, OSError):
            pass

    return apply


def _scrubbed_env(work_dir: str) -> dict[str, str]:
    """Minimal environment for the fallback path (bwrap uses --clearenv)."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": work_dir,
        "TMPDIR": work_dir,
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
    }


def run(
    argv: list[str],
    work_dir: str,
    timeout: float,
    *,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    fsize_bytes: int = DEFAULT_FSIZE_BYTES,
    nproc: int = DEFAULT_NPROC,
) -> subprocess.CompletedProcess:
    """Run `argv` (cwd = `work_dir`) under the resolved isolation policy and
    return the CompletedProcess. Raises subprocess.TimeoutExpired on timeout.
    Only `work_dir` is writable; everything else is best-effort locked down."""
    global _warned_fallback
    use_bwrap = MODE == "bwrap" or (MODE == "auto" and bwrap_works())

    cpu_seconds = int(timeout) + 1  # SIGXCPU backstop if wall-clock kill misses
    preexec = _rlimit_preexec(mem_bytes, cpu_seconds, fsize_bytes, nproc)

    if use_bwrap:
        return subprocess.run(
            _bwrap_argv(work_dir, argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=preexec,
        )

    if MODE == "auto" and not _warned_fallback:
        _warned_fallback = True
        warnings.warn(
            "bubblewrap sandbox unavailable (kernel/container forbids "
            "namespaces); executing reward code in a hardened subprocess "
            "WITHOUT filesystem/network isolation. Set PICOCHAT_SANDBOX=bwrap "
            "to require the sandbox.",
            RuntimeWarning,
            stacklevel=2,
        )
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=work_dir,
        env=_scrubbed_env(work_dir),
        preexec_fn=preexec,
    )
