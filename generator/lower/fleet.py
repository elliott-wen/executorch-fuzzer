"""fleet.py — lower the graphs in an oracle corpus, supervised.

Says what a worker is and what the jobs are; the process handling is
mobile.generator.supervisor. Imports no torch and no backend SDK — nothing here exports or
lowers anything, which is what keeps a compiler segfault from costing more than one graph.

The jobs are a listing of the oracle corpus, so they are exactly the graphs that exist —
there is no index space to fall out of step with what is on disk.

    from mobile.generator.lower import lower
    lower("tmp/oracle", "tmp/pte/portable", backend="portable", workers=64)
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

from mobile.generator.supervisor import journal, pool

#: seconds one graph may take before its worker is presumed hung. Lowering is far slower than
#: graph generation — seconds, not milliseconds — so this is much larger than the oracle
#: stage's, and still only catches a genuine hang.
GRAPH_TIMEOUT = 900.0

#: graphs a worker lowers before it is retired. Lower than the oracle stage's, because
#: repeated export+lower leaks and accumulates global state more aggressively than graph
#: building does.
MAX_GRAPHS_PER_WORKER = 32

RECORD_GLOB = "[0-9a-f][0-9a-f][0-9a-f]/*.json"

_REPO_PARENT = str(Path(__file__).resolve().parents[3])


def worker_argv(oracle_root, out, backend: str, quantize: bool,
                python: str | None = None) -> list[str]:
    """The command that runs one worker.

    `python` names the interpreter, because some backends live in their own virtualenv (the
    MediaTek SDK ships a cp310 wheel, CUDA has its own build). The supervisor spawns whatever
    argv says and does not care that it is a different install.
    """
    argv = [python or sys.executable, "-m", "mobile.generator.lower.worker",
            "--oracle", str(oracle_root), "--out", str(out), "--backend", backend]
    if quantize:
        argv.append("--quantize")
    return argv


def worker_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [_REPO_PARENT, env.get("PYTHONPATH", "")]))
    return env


def lower(oracle_root, out, backend: str = "portable", count: int | None = None,
          workers: int = 0, quantize: bool = False,
          python: str | None = None, verbose: bool = True) -> int:
    """Lower the graphs in `oracle_root` into `out`, at most `count` of them.

    Returns how many .pte records this run wrote. A token already lowered is skipped, so
    re-running after a fix only redoes what is missing — exact, because the token comes from
    the corpus rather than being re-derived.
    """
    from mobile.generator.oracle import store as oracle_store

    workers = workers or (os.cpu_count() or 1)
    tokens = oracle_store.iter_tokens(oracle_root)
    if count is None:
        count = sum(1 for _ in oracle_store.iter_tokens(oracle_root))
    else:
        tokens = itertools.islice(tokens, count)
    argv = worker_argv(oracle_root, out, backend, quantize, python)
    log = journal.Journal(out, count, verbose, label="lower")

    if verbose:
        print(f"[lower] {workers} workers, backend={backend}"
              f"{' +quantize' if quantize else ''} → {out} "
              f"({count:,} graphs from {oracle_root})", flush=True)
    try:
        pool.run(argv, tokens,
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
