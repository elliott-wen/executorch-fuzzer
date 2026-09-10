"""journal.py — record what happened to every job, and show progress while it happens.

The natural companion to `pool.run(..., on_result=journal)`: a Journal is a callable that
takes a Result and does the two things every batch stage wants — append it to a log, and
print a progress line now and then. It knows nothing about what the jobs were, so the same
one serves graph generation, lowering, or anything else run under the supervisor.

Outcomes go to `<out>/outcomes.tsv`, one line per job:

    <job>  <outcome>  <stage>  <detail>

`stage` is where the worker was when it finished or died, which for a crash or a timeout is
the only thing that says which part of the job was at fault. The parent is the only writer,
so there is no locking here and no per-worker file to merge afterwards.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from mobile.generator.supervisor.child import Result

OUTCOMES = "outcomes.tsv"
WORKER_STDERR = "worker-stderr.log"

#: seconds between progress lines
PROGRESS_SECONDS = 15.0


class Journal:
    """Records outcomes and prints progress. Called serially by the pool."""

    def __init__(self, out, total: int, verbose: bool = True, label: str = "stage") -> None:
        Path(out).mkdir(parents=True, exist_ok=True)
        self._file = open(Path(out) / OUTCOMES, "a")
        #: where workers' stderr goes — the only place a native abort message survives, since
        #: no Python handler runs for one. Shared by every worker: the messages are small and
        #: O_APPEND keeps them from interleaving.
        self.stderr = open(Path(out) / WORKER_STDERR, "a")
        self.outcomes: Counter = Counter()
        self._total, self._verbose, self._label = total, verbose, label
        self._started = self._last_report = time.time()

    def __call__(self, result: Result) -> None:
        detail = result.detail
        if result.outcome in ("crash", "timeout"):
            detail = f"in {result.stage}: {detail}"     # which stage killed it
        self._file.write(f"{result.job}\t{result.outcome}\t{result.stage}\t{detail}\n")
        self._file.flush()
        self.outcomes[result.outcome] += 1
        now = time.time()
        if self._verbose and now - self._last_report >= PROGRESS_SECONDS:
            self._last_report = now
            print(self.progress(), flush=True)

    def progress(self) -> str:
        done = sum(self.outcomes.values())
        if not done:
            return f"[{self._label}] starting"
        elapsed = time.time() - self._started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = f"{max(0, self._total - done) / rate / 60:.0f}m" if rate else "?"
        ready = self.outcomes["ready"]
        rest = " ".join(f"{k}={v}" for k, v in sorted(self.outcomes.items()) if k != "ready")
        return (f"[{self._label}] {done:,}/{self._total:,}  {ready:,} ready ({ready / done:.0%})"
                f"  {rate:.0f}/s  eta {eta}" + (f"  | {rest}" if rest else ""))

    def close(self) -> None:
        self._file.close()
        self.stderr.close()


def summarize(out, record_glob: str, top_reasons: int = 15) -> str:
    """A report on a stage's output directory: what it holds, and how the runs went.

    The record count comes from the files themselves (`record_glob`), so it is right no
    matter how many runs built the directory; the outcome mix comes from outcomes.tsv, which
    accumulates across runs and so counts a retried job more than once.
    """
    held = sum(1 for _ in Path(out).glob(record_glob))
    outcomes: Counter = Counter()
    reasons: Counter = Counter()
    path = Path(out) / OUTCOMES
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            fields = line.split("\t")
            if len(fields) != 4:
                continue
            _job, outcome, _stage, detail = fields
            outcomes[outcome] += 1
            if outcome not in ("ready", "done"):
                reasons[f"{outcome}: {detail[:70]}"] += 1
    logged = sum(outcomes.values())
    lines = [f"{out}", f"  records: {held:,}", f"  outcomes logged: {logged:,}"]
    for outcome, n in outcomes.most_common():
        lines.append(f"    {outcome:<14} {n:>10,}  {n / logged:>6.1%}" if logged
                     else f"    {outcome:<14} {n:>10,}")
    if reasons:
        lines.append("  top reasons:")
        for reason, n in reasons.most_common(top_reasons):
            lines.append(f"    {n:>8,}  {reason}")
    return "\n".join(lines)
