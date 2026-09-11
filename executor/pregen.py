"""pregen.py — pre-generate a corpus of jobs to disk (generate + export + serialize).

A graph is a pure function of (seed, index) — index is a plain non-negative integer, no
producer/worker identity is baked into it. This means:
  - however many processes you run to generate a range of indices is a pure scheduling
    decision (see pregen_fleet.py's shared-counter dispatch) — it has no effect on what
    graph index `i` is;
  - two backends exporting the SAME oracle index get the exact same graph, joinable by
    job_id, for free.

Three modes share one per-index processing loop (run_worker):
  oracle  (step_oracle) — graph gen + emit + eager reference. Backend-agnostic, run once.
  export  (step_export) — reads one oracle record + torch.export + backend lower. Run once
                          per backend against a (possibly still-growing) oracle corpus.
  full    (step_full)   — oracle then export inline, one process, no separate oracle corpus
                          (convenient for a quick single-backend smoke test).

Dispatch: a worker reads batches ("<start> <count>", one per line) from stdin and prints
"__READY__" after finishing each one so the fleet coordinator (pregen_fleet.py) sends more —
or, run directly (no fleet), a worker just processes one fixed [start, start+count) range.
Before doing any work for index `i`, a worker checks whether its output file already exists
and skips instantly if so — this one check is what makes both "restart later" and "redo a
crashed worker's whole in-flight batch" safe and cheap, with no persisted counter or
per-worker skip-list needed.

Either the eager run or the lowering can *hard-crash* on some graphs — a bug worth keeping.
We don't try to survive it: a marker file for the CURRENT index is written BEFORE the risky
call (build_oracle/build_export/build_job) and left behind if the process dies, purely as a
debugging artifact (mobile/gen/export/job.py) — nothing in the fleet's control flow parses it
any more; a crashed/hung worker is just killed and its in-flight batch re-dispatched to a
fresh one (see pregen_fleet.py), safe because of the resume check above.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import random
import signal
import sys
from collections import Counter
from pathlib import Path

from mobile.gen.graph import build_graph
from mobile.gen.ops import load_runnable_opnodes
from mobile.gen.export import build_oracle, build_export, get_backend, available_backends
from mobile.executor import protocol as P
from mobile.executor import corpus as C


# ── graph derivation: a pure function of (seed, index) ──────────────────────────────

def _sched_order(ops) -> list[int]:
    """A fixed shuffle of op indices, shared by every index/process (seeded independent of
    `seed` on purpose — the SHAPE of the schedule doesn't change with the run's seed, only
    where in it a given index falls, via `seed`'s offset below)."""
    order = list(range(len(ops)))
    random.Random(0xC0FFEE).shuffle(order)
    return order


def seed_op_for(ops, sched_order: list[int], seed: int, index: int):
    """The op `index` is scheduled to seed its graph with. Every op occupies its guaranteed
    seed slot an equal number of times across the whole index space (round-robin over the
    fixed global shuffle) instead of uniform-then-feasibility-filtered picks (which
    structurally starve constraint-heavy ops) — a pure function of (seed, index)."""
    offset = seed % len(ops)
    return ops[sched_order[(offset + index) % len(ops)]]


def rng_for(seed: int, index: int) -> random.Random:
    return random.Random(seed * 1_000_003 + index * 2_654_435_761 + 1)


def build_graph_for(ops, sched_order: list[int], seed: int, index: int, nodes: int,
                    leaf_prob: float, out_alias_prob: float):
    """(seed, index) -> (graph, seed_op), retried up to 8x (generate() is stochastic; the
    seed_op is fixed across retries so the coverage intent holds). `graph` is None if no
    valid DAG came out after all retries."""
    seed_op = seed_op_for(ops, sched_order, seed, index)
    rng = rng_for(seed, index)
    graph = None
    for _ in range(8):
        graph = build_graph(rng, ops, nodes, leaf_prob, seed_op=seed_op,
                            out_alias_prob=out_alias_prob)
        if graph is not None:
            break
    return graph, seed_op


def replay_index(seed: int, index: int, nodes: int = 8, leaf_prob: float = 0.3,
                 out_alias_prob: float = 0.1):
    """Regenerate the EXACT graph a given (seed, index) produced, returning (graph, src).
    Generation is a pure function of (seed, index) + these params, so a failure captured in
    _fails/<index>.py can be re-derived and inspected without re-running the whole fleet.
    `src` is None if the graph was un-emittable (an 'emit'-stage failure)."""
    ops = load_runnable_opnodes()
    sched_order = _sched_order(ops)
    graph, _ = build_graph_for(ops, sched_order, seed, index, nodes, leaf_prob, out_alias_prob)
    if graph is None:
        return None, None
    try:
        src = graph.emit(seed=seed)
    except Exception:
        src = None
    return graph, src


# ── diagnostics: one file per failed graph, keyed on index (never gates control flow) ──

def _record_fail(out: str, index: int, seed_op: str, stage: str, detail: str,
                 graph=None, src: str | None = None) -> None:
    fails_dir = Path(out) / "_fails"
    fails_dir.mkdir(parents=True, exist_ok=True)
    header = (f"# stage:   {stage}\n"
              f"# job_id:  {index}\n"
              f"# seed_op: {seed_op}\n"
              f"# reason:  {detail}\n")
    if src is not None:
        body = src
    elif graph is not None:
        body = f"# (un-emittable — graph description only)\n# {graph.describe()}\n"
    else:
        body = ""
    try:
        (fails_dir / f"{index}.py").write_text(header + body)
    except OSError:
        pass                                             # a failed diag write never sinks a run


def _crash_marker(out: str, slot: str) -> Path:
    d = Path(out) / "_crashes"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{slot}.py"


def _already_written(out: str, ext: str, index: int) -> bool:
    return C.job_file(out, str(index), ext=ext).exists()


# ── per-index steps ──────────────────────────────────────────────────────────────────

def step_oracle(out: str, ops, sched_order: list[int], seed: int, index: int, nodes: int,
                leaf_prob: float, out_alias_prob: float, marker: Path) -> str:
    """One oracle-stage attempt at global index `index`: graph gen + emit + eager. Returns
    the outcome stage ("ready"|"gen_fail"|"emit"|"eager"|"build_raise"). Writes
    <out>/<bucket>/<index>.{oracle,py} on success."""
    job_id = str(index)
    graph, seed_op = build_graph_for(ops, sched_order, seed, index, nodes, leaf_prob, out_alias_prob)
    if graph is None:
        return "gen_fail"                                # generator couldn't build a valid DAG
    try:
        src = graph.emit(seed=seed)
    except Exception as e:
        _record_fail(out, index, seed_op.op_name, "emit",
                     f"emit raised {type(e).__name__}: {e}", graph=graph)
        return "emit"
    if src is None:
        _record_fail(out, index, seed_op.op_name, "emit",
                     "emit returned None (un-emittable arg / no schema)", graph=graph)
        return "emit"
    marker.write_text(f"# crashed in build_oracle (eager run) — job_id {job_id}\n{src}")
    try:
        result = build_oracle(src)                       # eager run; may HARD-CRASH
    except Exception as e:
        _record_fail(out, index, seed_op.op_name, "build_raise",
                     f"{type(e).__name__}: {e}", src=src)
        return "build_raise"
    if result.status != "READY":
        _record_fail(out, index, seed_op.op_name, result.stage, result.detail, src=src)
        return result.stage
    frames = P.encode_oracle(job_id, result.inputs, result.eager, desc=graph.describe())
    C.write_job(out, job_id, frames, src=src, ext="oracle")
    return "ready"


def step_export(oracle_dir: str, out: str, backend: str, quantize: bool, index: int,
                marker: Path) -> str:
    """One export-stage attempt at global index `index`: read that index's oracle record (if
    it exists yet — an oracle still being generated just means some indices aren't there,
    a plain "no_oracle" skip, not an error) and lower it for `backend`. Returns the outcome
    stage. Writes <out>/<bucket>/<index>.{job,py} on success — same on-disk shape as
    build_job's, so broker/feed/et_runner/compare need no changes."""
    job_id = str(index)
    opath = C.job_file(oracle_dir, job_id, ext="oracle")
    if not opath.exists():
        return "no_oracle"
    _job_id, desc, inputs, eager = P.decode_oracle(C.read_job(opath))
    src = C.job_file(oracle_dir, job_id, ext="py").read_text()
    marker.write_text(f"# crashed in build_export (export/lower) — job_id {job_id}\n{src}")
    try:
        result = build_export(src, inputs, eager, backend, quantize)  # may HARD-CRASH
    except Exception as e:
        _record_fail(out, index, "", "build_raise", f"{type(e).__name__}: {e}", src=src)
        return "build_raise"
    if result.status != "READY":
        _record_fail(out, index, "", result.stage, result.detail, src=src)
        return result.stage
    _write_export_result(out, job_id, desc, backend, result, src)
    return "ready"


def step_full(out: str, ops, sched_order: list[int], seed: int, index: int, nodes: int,
             leaf_prob: float, out_alias_prob: float, backend: str, quantize: bool,
             marker: Path) -> str:
    """One full-pipeline attempt at global index `index`: graph gen + emit + eager + export +
    lower, all in this process (no separate oracle corpus). Returns the outcome stage."""
    job_id = str(index)
    graph, seed_op = build_graph_for(ops, sched_order, seed, index, nodes, leaf_prob, out_alias_prob)
    if graph is None:
        return "gen_fail"
    try:
        src = graph.emit(seed=seed)
    except Exception as e:
        _record_fail(out, index, seed_op.op_name, "emit",
                     f"emit raised {type(e).__name__}: {e}", graph=graph)
        return "emit"
    if src is None:
        _record_fail(out, index, seed_op.op_name, "emit",
                     "emit returned None (un-emittable arg / no schema)", graph=graph)
        return "emit"
    marker.write_text(f"# crashed in build_job (eager run or lowering) — job_id {job_id}\n{src}")
    try:
        oracle = build_oracle(src)
        result = oracle if oracle.status != "READY" else build_export(
            src, oracle.inputs, oracle.eager, backend, quantize)
    except Exception as e:
        _record_fail(out, index, seed_op.op_name, "build_raise",
                     f"{type(e).__name__}: {e}", src=src)
        return "build_raise"
    if result.status != "READY":
        _record_fail(out, index, seed_op.op_name, result.stage, result.detail, src=src)
        return result.stage
    _write_export_result(out, job_id, graph.describe(), backend, result, src)
    return "ready"


def _write_export_result(out: str, job_id: str, desc: str, backend: str, result, src: str) -> None:
    """Shared tail of step_export/step_full: annotate + write a READY ExportResult in the
    pushjob format (unchanged from build_job's on-disk shape)."""
    delegated = {"ops": result.delegated_ops, "non": result.non_delegated_ops,
                "calls": result.delegate_calls}
    frames = P.encode_pushjob(job_id, result.pte, result.inputs, result.eager, desc=desc,
                              user_pos=result.user_pos, delegated=delegated)
    src_annotated = (f"# delegated: backend={backend} ops={result.delegated_ops} "
                     f"non_delegated={result.non_delegated_ops} "
                     f"delegate_calls={result.delegate_calls}\n{src}")
    C.write_job(out, job_id, frames, src=src_annotated, ext="job")


# ── shared batch-dispatch loop ───────────────────────────────────────────────────────

def _stdin_batches():
    """Yield (start, count) batches read from stdin, one per line — see pregen_fleet.py's
    coordinator. Only asks for MORE after the caller has finished processing the current
    batch (the `yield` suspends here until the consuming `for` loop comes back for the next
    one), so "__READY__" always means "I'm done with what you last gave me"."""
    for line in sys.stdin:
        line = line.strip()
        if not line or line == "STOP":
            return
        start, count = (int(x) for x in line.split())
        yield start, count
        print("__READY__", flush=True)


def run_worker(out: str, out_ext: str, slot: str, marker: Path, step_fn, batches) -> int:
    """Shared driver for all three modes: resume-skip, per-index heartbeat, stats
    accumulation/flush, graceful shutdown. `step_fn(index) -> stage` and `batches` (an
    iterable of (start, count)) carry everything mode-specific."""
    stats_dir = Path(out) / "_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_file = stats_dir / f"{slot}.json"
    outcomes: dict[str, str] = {}
    if stats_file.exists():
        try:
            outcomes = dict(json.loads(stats_file.read_text()).get("outcomes", {}))
        except (ValueError, OSError):
            pass                                         # corrupt/partial file → fresh tally

    def _flush_stats() -> None:
        tmp = stats_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"slot": slot, "stages": dict(Counter(outcomes.values())),
                                   "outcomes": outcomes}))
        tmp.replace(stats_file)                          # atomic — the fleet may read concurrently

    def _graceful(*_):
        try:
            _flush_stats()
        except Exception:
            pass
        marker.unlink(missing_ok=True)
        os._exit(0)
    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    written = 0
    for start, count in batches:
        for index in range(start, start + count):
            if _already_written(out, out_ext, index):
                continue                                 # resume / crash-requeue: idempotent skip
            print(f"__HB__ {index}", flush=True)
            outcomes[str(index)] = step_fn(index)
            if outcomes[str(index)] == "ready":
                written += 1
        _flush_stats()                                   # checkpoint after every batch

    marker.unlink(missing_ok=True)                        # finished cleanly → no crash
    ready = sum(1 for v in outcomes.values() if v == "ready")
    print(f"[pregen {slot}] done: {written} written this run ({ready} ready total) → {out}/",
          flush=True)
    return 0


# ── CLI entry (see pregen.py / mobile/__main__.py) ───────────────────────────────────

def run(mode: str, out: str, oracle: str, seed: int, nodes: int, leaf_prob: float,
       out_alias_prob: float, backend: str, quantize: bool, slot: str,
       start: int, count: int, use_stdin: bool) -> int:
    if mode in ("export", "full") and get_backend(backend) is None:
        print(f"[pregen {slot}] unknown backend {backend!r}; "
              f"available here: {', '.join(available_backends())}", flush=True)
        return 1

    ops = sched_order = None
    if mode in ("oracle", "full"):
        print(f"[pregen {slot}] loading op set ...", flush=True)
        ops = load_runnable_opnodes()
        sched_order = _sched_order(ops)

    marker = _crash_marker(out, slot)
    if mode == "oracle":
        out_ext = "oracle"
        step_fn = lambda index: step_oracle(out, ops, sched_order, seed, index, nodes,
                                            leaf_prob, out_alias_prob, marker)
    elif mode == "export":
        out_ext = "job"
        step_fn = lambda index: step_export(oracle, out, backend, quantize, index, marker)
    elif mode == "full":
        out_ext = "job"
        step_fn = lambda index: step_full(out, ops, sched_order, seed, index, nodes,
                                          leaf_prob, out_alias_prob, backend, quantize, marker)
    else:
        print(f"[pregen {slot}] unknown mode {mode!r} (have: oracle, export, full)", flush=True)
        return 1

    batches = _stdin_batches() if use_stdin else [(start, count)]
    return run_worker(out, out_ext, slot, marker, step_fn, batches)
