"""fleet.py — generate a batch of graphs, supervised.

All the process handling lives in mobile.generator.supervisor; this file only says what a
worker is and what the jobs are. It imports no torch, and nothing here ever runs a generated
graph — that is what keeps a segfault in a worker from being anything worse than one lost
graph.

The jobs ARE the tokens, minted here. Generation is append-only: a corpus is however many
records it holds, so there is no range to resume and no gap to fill — running again simply
adds more graphs.

    from mobile.generator.oracle import Params, generate
    generate("tmp/oracle", params=Params(), count=1_000_000, workers=96)
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from mobile.generator.oracle.params import Params
from mobile.generator.supervisor import journal, pool

#: seconds one graph may take before its worker is presumed hung. Generous: the slowest
#: legitimate graphs run well under a second, so only real hangs hit it.
GRAPH_TIMEOUT = 120.0

#: graphs a worker makes before it is retired. Crash isolation covers processes that die;
#: this covers ones that quietly degrade — torch accumulates global state across many graphs.
MAX_GRAPHS_PER_WORKER = 64

RECORD_GLOB = "[0-9a-f][0-9a-f][0-9a-f]/*.oracle"

_REPO_PARENT = str(Path(__file__).resolve().parents[3])


def worker_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [_REPO_PARENT, env.get("PYTHONPATH", "")]))
    return env


def generate(out, params: Params | None = None, count: int = 10_000,
             workers: int = 0, verbose: bool = True) -> int:
    """Attempt `count` graphs into `out` across `workers` processes.

    Returns how many records this run wrote — fewer than `count`, since a graph PyTorch
    rejects produces no record. Running again adds more graphs rather than filling gaps:
    every job is a fresh token, so there is nothing to collide with and nothing to resume.
    To reach a target size, run until the corpus holds enough.
    """
    params = params or Params()
    workers = workers or (os.cpu_count() or 1)
    # Absolute, because workers run with cwd set to the repo's parent so `import mobile`
    # resolves. A relative --out would land beside the repo instead of where the caller
    # meant, splitting the records away from the outcomes log written here.
    out = Path(out).resolve()
    argv = [sys.executable, "-m", "mobile.generator.oracle.worker",
            "--out", str(out), *params.to_argv()]
    log = journal.Journal(out, count, verbose, label="oracle")

    if verbose:
        print(f"[oracle] {workers} workers → {out} ({count:,} graphs)", flush=True)
    try:
        pool.run(argv, (uuid.uuid4().hex for _ in range(count)),
                 workers=workers, on_result=log, timeout=GRAPH_TIMEOUT,
                 max_jobs_per_child=MAX_GRAPHS_PER_WORKER,
                 env=worker_env(), cwd=_REPO_PARENT, stderr=log.stderr)
        if verbose:
            print(log.progress(), flush=True)
        return log.outcomes["ready"]
    finally:
        log.close()


def summarize(out, top_reasons: int = 15) -> str:
    return journal.summarize(out, RECORD_GLOB, top_reasons)
