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

#: Backends that leak fast enough to need retiring sooner than that. MEASURED, per backend —
#: a number here is a workaround for a leak somewhere in that backend's AoT stack, so it
#: belongs with a note saying what was seen rather than being tuned by feel.
#:
#: cadence: a worker holds ~400 MB flat for its first ~10 graphs, then climbs steeply —
#: 445 MB at job 10, 501 at 11, 694 at 12, 1,462 at 14. Object COUNT grows only linearly
#: (~3.5k/job, mostly FrameSummary/StackSummary), so it is retained tracebacks holding whole
#: graph modules alive rather than a runaway container. At 32 jobs x 96 workers that reached
#: ~14 GB within 100 s and stalled the machine; 8 keeps a worker under ~450 MB.
RECYCLE_AFTER = {"cadence": 8}

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


def _require_backend(argv, backend: str) -> None:
    """Probe the backend in one throwaway process before opening the pool.

    Without this an unavailable backend is discovered the hard way: every worker exits
    immediately, the supervisor's barren-spawn guard retires each slot after a few tries,
    and the run ends with hundreds of anonymous `crash` outcomes and no .pte — saying only
    that something died, never that the SDK simply is not installed. One probe costs a
    couple of seconds and turns that into a sentence. It also keeps the parent free of torch
    and of the backend's SDK, which is the whole point of doing it in a subprocess.
    """
    import subprocess

    probe = subprocess.run(argv + ["--check"], capture_output=True, text=True,
                           env=worker_env(), cwd=_REPO_PARENT, timeout=300)
    if probe.returncode != 0:
        detail = (probe.stderr or "").strip().splitlines()
        raise RuntimeError(f"backend {backend!r} is not usable in this install"
                           + (f" — {detail[-1]}" if detail else ""))


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
    # Absolute for both: workers run with cwd set to the repo's parent (so `import mobile`
    # resolves), and a relative path would resolve there rather than where the caller meant.
    oracle_root, out = Path(oracle_root).resolve(), Path(out).resolve()
    tokens = oracle_store.iter_tokens(oracle_root)
    if count is None:
        count = sum(1 for _ in oracle_store.iter_tokens(oracle_root))
    else:
        tokens = itertools.islice(tokens, count)
    argv = worker_argv(oracle_root, out, backend, quantize, python)
    _require_backend(argv, backend)
    log = journal.Journal(out, count, verbose, label="lower")

    if verbose:
        print(f"[lower] {workers} workers, backend={backend}"
              f"{' +quantize' if quantize else ''} → {out} "
              f"({count:,} graphs from {oracle_root})", flush=True)
    try:
        pool.run(argv, tokens,
                 workers=workers, on_result=log, timeout=GRAPH_TIMEOUT,
                 max_jobs_per_child=RECYCLE_AFTER.get(backend, MAX_GRAPHS_PER_WORKER),
                 env=worker_env(), cwd=_REPO_PARENT, stderr=log.stderr)
        if verbose:
            print(log.progress(), flush=True)
        return log.outcomes["ready"]
    finally:
        log.close()


def summarize(out, top_reasons: int = 15) -> str:
    return journal.summarize(out, RECORD_GLOB, top_reasons)
