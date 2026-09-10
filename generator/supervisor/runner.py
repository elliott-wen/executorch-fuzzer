"""runner.py — the child half: read jobs from stdin, report what happened to stdout.

A worker program is a `handle` function plus one call to `serve`. Everything about being a
supervised process — the protocol, the flushing, turning an unexpected exception into an
outcome instead of an exit — lives here, so a worker contains only its own work.

    def handle(job, enter):
        enter("parse")
        ...
        return "ok", ""

    serve(handle)

Nothing may be written to stdout except protocol lines: it is the channel the parent reads.
Logging, warnings and native crash messages belong on stderr, which the parent redirects
somewhere durable.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator

from mobile.generator.supervisor import protocol


def _emit(line: str) -> None:
    """Write one protocol line and flush it. Flushing per line is the point: a process that
    is about to be killed must not take its progress with it."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def enter(stage: str) -> None:
    """Announce the stage about to start, so a crash inside it is attributable."""
    _emit(protocol.enter_line(stage))


def _jobs() -> Iterator[str]:
    """Jobs from stdin, one per line, until the parent closes the pipe."""
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        job = line.strip()
        if job:
            yield job


def serve(handle: Callable[[str, Callable[[str], None]], tuple[str, str]]) -> None:
    """Run jobs until stdin closes.

    `handle(job, enter) -> (outcome, detail)`. An exception escaping `handle` becomes an
    "error" outcome rather than ending the process: the job failed, the worker did not, and
    the parent should keep using it. Only a signal or a native abort ends a worker, and that
    is exactly the case the parent is watching for.
    """
    for job in _jobs():
        try:
            outcome, detail = handle(job, enter)
        except Exception as exc:                      # noqa: BLE001 — a bad job, not a bad worker
            outcome, detail = "error", f"{type(exc).__name__}: {exc}"
        _emit(protocol.done_line(outcome, detail))
