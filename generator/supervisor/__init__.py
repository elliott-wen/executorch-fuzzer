"""supervisor — run work in disposable child processes and survive losing them.

The problem it solves: the work this project does can take its process down. Running a
generated graph eagerly, lowering it, executing a `.pte` — each can segfault or corrupt the
heap rather than raise, and no Python handler runs when it does. So the work happens in
children and the parent only supervises.

    parent (this package)              child (runner.serve)
      spawn ─────────────────────────▶ load whatever is expensive, once
      job ───────────────────────────▶ enter("eager") …
                                       ← enter  eager
                                       ← done   ready
      next job ─────────────────────▶  …

The parent never imports torch, never runs a graph, and never lowers anything. That is the
whole guarantee, and it is worth keeping: it is why a crash costs one job rather than a run.

Losing a child is ordinary, not exceptional — measured at roughly one job in 250 — so it is
a return value (`Result.fatal`), never an exception. Because jobs are dispatched one at a
time and acknowledged, the parent always knows exactly which job was in flight; and because
the child announces each stage it enters, the parent knows which part of that job did it.

  protocol  the wire format, in one place so the two halves cannot drift
  child     the parent's handle on one process: submit a job, get a Result
  pool      N workers over a job stream, replacing children that die or age out
  runner    the child half: serve(handle)
  journal   record every outcome and show progress — the usual `on_result`
"""

from __future__ import annotations

from mobile.generator.supervisor.child import Child, Result
from mobile.generator.supervisor.journal import Journal
from mobile.generator.supervisor.pool import DEFAULT_MAX_JOBS_PER_CHILD, DEFAULT_TIMEOUT, run
from mobile.generator.supervisor.runner import serve

__all__ = [
    "Child", "Result", "Journal", "run", "serve",
    "DEFAULT_TIMEOUT", "DEFAULT_MAX_JOBS_PER_CHILD",
]
