"""protocol.py — the line protocol between a supervisor and its worker processes.

Parent to child: one line per job, the job payload verbatim.
Child to parent: one line per event, tab-separated.

    enter <stage>              starting <stage>; nothing past here is guaranteed
    done  <outcome> <detail>   the job finished with this outcome

`enter` is what makes a crash attributable. A process killed by a segfault sends nothing
further and gets no chance to write a marker file, so the last stage it announced is the
stage that killed it. That matters because the risky steps of a job fail in quite different
ways — running a graph eagerly, lowering it, executing it — and "died somewhere" is not a
finding anyone can act on.

Everything about the wire format lives here, so the parent and the child cannot drift.
"""

from __future__ import annotations

ENTER = "enter"
DONE = "done"

_STAGE_LIMIT = 40
_DETAIL_LIMIT = 200


def clean(text: object, limit: int = _DETAIL_LIMIT) -> str:
    """Collapse a value to one tab-free line, short enough to log."""
    return " ".join(str(text).split())[:limit]


def enter_line(stage: str) -> str:
    return f"{ENTER}\t{clean(stage, _STAGE_LIMIT)}"


def done_line(outcome: str, detail: str = "") -> str:
    return f"{DONE}\t{clean(outcome, _STAGE_LIMIT)}\t{clean(detail)}"


def parse(line: str) -> tuple[str, list[str]] | None:
    """(kind, fields) for one complete protocol line, or None if it isn't one.

    A line with no trailing newline is rejected rather than parsed: the child writes whole
    lines and flushes, so a partial line means it died mid-write.
    """
    if not line.endswith("\n"):
        return None
    head, *fields = line.rstrip("\n").split("\t")
    if head == ENTER and len(fields) == 1:
        return ENTER, fields
    if head == DONE and len(fields) == 2:
        return DONE, fields
    return None
