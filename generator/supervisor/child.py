"""child.py — the parent's handle on one worker process.

A `Child` is fed one job at a time and answers with a `Result`. It owns exactly two
concerns: keeping the pipe protocol honest, and turning "the process is gone" into an
ordinary return value rather than an exception the caller has to catch everywhere.

Reading is done on the raw file descriptor rather than through `proc.stdout`. A buffered
reader can hold a line that `select` will never announce as pending, which would look
exactly like a hang; here `select` is the whole truth, and the line splitting is ours.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    """What became of one job."""

    job: str
    outcome: str      #: the child's own outcome, or "crash" / "timeout" if it never answered
    detail: str
    stage: str        #: the last stage the child announced — where a crash or hang happened
    fatal: bool       #: the child is gone; the caller must not submit to it again


def _exit_text(code: int | None) -> str:
    """'signal 11 (SIGSEGV)' rather than 'exit -11' — the signal is the whole diagnosis."""
    if code is None:
        return "still running"
    if code >= 0:
        return f"exit {code}"
    try:
        return f"signal {-code} ({signal.Signals(-code).name})"
    except ValueError:
        return f"signal {-code}"


class _Lines:
    """Whole lines from a raw fd, with a deadline. Returns "" at EOF, None at the deadline."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buffer = b""

    def readline(self, deadline: float) -> str | None:
        while True:
            end = self._buffer.find(b"\n")
            if end >= 0:
                line, self._buffer = self._buffer[:end + 1], self._buffer[end + 1:]
                return line.decode("utf-8", "replace")
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self._fd], [], [], remaining)[0]:
                return None
            chunk = os.read(self._fd, 65536)
            if not chunk:
                return ""      # EOF; any trailing partial line died with the process
            self._buffer += chunk


class Child:
    """One worker subprocess. Not thread-safe: one owner, one job at a time."""

    def __init__(self, argv, env=None, cwd=None, stderr=None) -> None:
        self._proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
            env=env, cwd=cwd, bufsize=0,
        )
        self._lines = _Lines(self._proc.stdout.fileno())
        self.jobs_done = 0

    def submit(self, job: str, timeout: float) -> Result:
        """Run one job. Never raises: a dead or hung child comes back as a Result."""
        from mobile.generator.supervisor import protocol

        stage = "start"
        try:
            self._proc.stdin.write(f"{job}\n".encode())
        except (BrokenPipeError, ValueError, OSError):
            return self._died(job, stage, "worker gone before the job was sent")

        deadline = time.monotonic() + timeout
        while True:
            line = self._lines.readline(deadline)
            if line is None:
                self.close()
                return Result(job, "timeout", f"no result within {timeout:.0f}s", stage, True)
            if line == "":
                return self._died(job, stage, "worker died")
            message = protocol.parse(line)
            if message is None:
                continue                      # stray stdout: ignore it, the deadline still holds
            kind, fields = message
            if kind == protocol.ENTER:
                stage = fields[0]
            else:
                self.jobs_done += 1
                return Result(job, fields[0], fields[1], stage, False)

    def close(self) -> None:
        """Shut the child down and reap it. Safe to call more than once."""
        self._reap()

    def _died(self, job: str, stage: str, why: str) -> Result:
        return Result(job, "crash", f"{why} ({_exit_text(self._reap())})", stage, True)

    def _reap(self) -> int | None:
        for stream in (self._proc.stdin, self._proc.stdout):
            try:
                stream.close()
            except OSError:
                pass
        if self._proc.poll() is None:
            self._proc.kill()
        try:
            return self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return None
