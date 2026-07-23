# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sandboxed executor for model-generated test candidates (issue #141).

Runs each (setup_code, program, test_assert) job in a persistent worker
subprocess and enforces the wall-clock limit from the parent with a hard
SIGKILL. This replaces the in-process SIGALRM approach, which a candidate
could escape simply by having a ``try/except`` around its hot loop (the alarm
fires once, the candidate's own handler swallows it, and the eval hangs — the
vet-cond-v2 failure). A kill from outside the process cannot be caught,
ignored, or swallowed by anything the candidate does.

Design:

* The worker is ``python -u -m kelp.cli._test_runner`` — this module doubles as
  the worker entrypoint. It imports only the stdlib, so respawns are cheap and
  it never touches JAX/TPU state.
* Protocol: one JSON object per line on stdin, one ``"1"``/``"0"`` line on
  stdout per job. The worker re-points fds 0/1 at ``os.devnull`` before running
  any candidate code (keeping private dups for the protocol), so a candidate
  that prints or reads stdin cannot corrupt the protocol or block on input.
* On timeout, crash (``os._exit``, segfault), or protocol garbage, the parent
  SIGKILLs and reaps the worker, reports the test as failed, and respawns
  lazily on the next call.
* Worker state (imported modules) persists across jobs for speed; each job gets
  a fresh namespace dict, same isolation level the in-process version had.

Unix only (uses ``select`` on pipes), like the SIGALRM version before it.
"""

import json
import logging
import os
import select
import subprocess
import sys

logger = logging.getLogger(__name__)


class SubprocessTestRunner:
    """Runs candidate/test pairs in a kill-on-timeout worker subprocess."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    def _ensure_worker(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "kelp.cli._test_runner"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        return self._proc

    def _kill_worker(self) -> None:
        if self._proc is None:
            return
        self._proc.kill()
        self._proc.wait()
        self._proc = None

    def run(self, program: str, test_assert: str, setup_code: str = "", timeout_s: float = 5.0) -> bool:
        """Execute ``setup_code``, ``program``, then ``test_assert`` in the worker.

        Returns True iff all three executed without raising within ``timeout_s``
        seconds (``timeout_s <= 0`` disables the limit). Any timeout, crash, or
        protocol failure counts as a failed test, never an exception here.
        """
        proc = self._ensure_worker()
        job = json.dumps({"setup": setup_code, "program": program, "test": test_assert})
        try:
            proc.stdin.write((job + "\n").encode())
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            # Worker died between jobs; one respawn-and-retry, then give up.
            self._kill_worker()
            proc = self._ensure_worker()
            try:
                proc.stdin.write((job + "\n").encode())
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._kill_worker()
                return False

        timeout = timeout_s if timeout_s > 0 else None
        readable, _, _ = select.select([proc.stdout], [], [], timeout)
        if not readable:
            logger.debug("Test execution timed out after %.1fs; killing worker", timeout_s)
            self._kill_worker()
            return False

        reply = proc.stdout.readline()
        if reply not in (b"1\n", b"0\n"):
            # EOF (worker crashed mid-job, e.g. os._exit) or garbage.
            self._kill_worker()
            return False
        return reply == b"1\n"

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            self._kill_worker()
        self._proc = None


_default_runner: SubprocessTestRunner | None = None


def default_runner() -> SubprocessTestRunner:
    """Process-wide shared runner (one worker amortized across all tests)."""
    global _default_runner
    if _default_runner is None:
        _default_runner = SubprocessTestRunner()
    return _default_runner


def _worker_main() -> int:
    # Keep private handles on the real stdio for the protocol, then point fds
    # 0/1 at devnull so candidate code can neither corrupt replies via print()
    # nor block reading stdin (input() sees immediate EOF instead).
    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)

    for line in proto_in:
        try:
            job = json.loads(line)
            ok = False
            namespace: dict = {}
            try:
                if job["setup"]:
                    exec(job["setup"], namespace)  # noqa: S102 - eval worker exists to execute generated code
                exec(job["program"], namespace)  # noqa: S102
                exec(job["test"], namespace)  # noqa: S102
                ok = True
            except BaseException:  # noqa: BLE001 - any raise (incl. SystemExit) is a failed test
                ok = False
            proto_out.write(b"1\n" if ok else b"0\n")
            proto_out.flush()
        except (json.JSONDecodeError, KeyError, OSError):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(_worker_main())
