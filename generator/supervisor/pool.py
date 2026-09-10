"""pool.py — run a stream of jobs across N worker processes.

One thread per worker, each holding one long-lived child and feeding it jobs one at a time.
Threads are the right tool because a thread here does nothing but wait on its child; the
work is all in the subprocesses, so the GIL is never contended.

Two things end a child's life, and they are different problems:

  it died      a job took the process down. Only the in-flight job is lost — the parent
               knows which one, because it is the one that never came back — so it is
               recorded as a crash and the next job gets a fresh process.
  it aged out  a child is retired after `max_jobs_per_child` jobs. Crash isolation protects
               against processes that DIE; recycling protects against ones that DEGRADE.
               Long-lived work here accumulates global state (an exir verifier that grew a
               module-level list on every lowering once slowed a whole run to a crawl) and
               leaks memory, neither of which announces itself.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable

from mobile.generator.supervisor.child import Child, Result

#: seconds one job may take before its worker is presumed hung and killed
DEFAULT_TIMEOUT = 300.0

#: jobs a child may run before it is retired and replaced
DEFAULT_MAX_JOBS_PER_CHILD = 64

#: consecutive freshly-spawned children that die on their very first job before we conclude
#: the worker itself is broken (a bad import, an unreadable path) rather than the jobs being
#: unlucky. A genuine poison job kills one child; five in a row is not bad luck.
_MAX_BARREN_SPAWNS = 5


class _Source:
    """The shared job stream, and the shared place results are reported.

    Two locks, because these are two unrelated things. Handing out a job has no reason to
    wait on a worker writing a result line, and `report` does real work under its lock — a
    write, a flush, sometimes a progress line.

    `take` needs a lock at all only because iterators are not thread-safe: concurrent
    `next()` on a generator raises, and on other iterators can silently skip or duplicate a
    job. It is held across `next()`, which is free for the finite, non-blocking sources this
    is used with (a range of indices, a listing of a corpus). A source that BLOCKS would
    serialize every worker behind it and would need a feeder thread and a queue instead.

    `on_result` is called serially, so callers need no locking of their own — which keeps
    the reporting side as simple as it looks.
    """

    def __init__(self, jobs: Iterable[str], on_result: Callable[[Result], None]) -> None:
        self._jobs = iter(jobs)
        self._on_result = on_result
        self._jobs_lock = threading.Lock()
        self._report_lock = threading.Lock()

    def take(self) -> str | None:
        with self._jobs_lock:
            return next(self._jobs, None)

    def report(self, result: Result) -> None:
        with self._report_lock:
            self._on_result(result)


def _session(source: _Source, spawn: Callable[[], Child], timeout: float,
             max_jobs: int) -> None:
    """One worker slot: keep a child alive and feed it jobs until the stream runs dry."""
    child: Child | None = None
    barren_spawns = 0
    try:
        while True:
            job = source.take()
            if job is None:
                return
            if child is None:
                child = spawn()
            result = child.submit(job, timeout)
            source.report(result)

            if result.fatal:
                barren_spawns = barren_spawns + 1 if child.jobs_done == 0 else 0
                child.close()
                child = None
                if barren_spawns >= _MAX_BARREN_SPAWNS:
                    return                      # the worker is broken, not the jobs
            elif child.jobs_done >= max_jobs:
                child.close()
                child = None
    finally:
        if child is not None:
            child.close()


def run(argv, jobs: Iterable[str], *, workers: int,
        on_result: Callable[[Result], None],
        timeout: float = DEFAULT_TIMEOUT,
        max_jobs_per_child: int = DEFAULT_MAX_JOBS_PER_CHILD,
        env=None, cwd=None, stderr=None) -> None:
    """Run every job in `jobs` across `workers` processes, reporting each to `on_result`.

    Returns once the stream is exhausted and every child has been shut down. `jobs` may be a
    generator — it is consumed lazily, so an unbounded stream is fine and the pace is set by
    how fast the workers finish.
    """
    source = _Source(jobs, on_result)
    spawn = lambda: Child(argv, env=env, cwd=cwd, stderr=stderr)  # noqa: E731
    threads = [threading.Thread(target=_session, args=(source, spawn, timeout,
                                                       max_jobs_per_child),
                                name=f"worker-{slot}", daemon=True)
               for slot in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
