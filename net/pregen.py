"""pregen.py — pre-generate a corpus of jobs to disk (generate + export + serialize).

Each READY graph is lowered to ExecuTorch and written to a corpus directory; pre-gen
makes a run reproducible and every graph inspectable on disk (corpus/<job_id>.py).

`build_job` runs the eager reference, then lowers (torch.export → to_executorch). Either
step can *hard-crash* on some graphs (the eager kernel or the ExecuTorch compiler) — a
bug worth keeping. We don't try to survive it: the in-flight graph is written to
corpus/_crashes/<id>.py BEFORE the risky call and removed on clean exit, so if the worker
dies that file is left behind. The fleet then RESPAWNS the worker after recording the
crashing index in corpus/_crashes/<id>.skip; on restart the worker SKIPS those indices
(and resumes past already-written jobs) so it goes past the crash instead of re-hitting it.

Run several pregen processes with distinct --id into the SAME --out for parallel, disjoint
generation (different ids ⇒ different filenames + disjoint graph space).
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import random
import signal
import time
import zlib
from collections import Counter
from pathlib import Path

from mobile.gen.graph import build_graph
from mobile.gen.ops import load_runnable_opnodes
from mobile.gen.export import build_job, get_backend, available_backends
from mobile.net import protocol as P
from mobile.net import corpus as C


def replay_index(producer_id: str, cur: int, nodes: int = 8, leaf_prob: float = 0.3,
                 seed: int = 0xC0FFEE, out_alias_prob: float = 0.1):
    """Regenerate the EXACT graph a given (producer_id, cur) produced, returning
    (graph, src). Generation is a pure function of (producer_id, cur) + these params,
    so a failure captured in _fails/<producer_id>_<cur>.py can be re-derived and
    inspected without re-running the whole fleet (the filename's cur is all you need).
    `src` is None if the graph was un-emittable (an 'emit'-stage failure). Mirrors
    run_pregen's inner loop."""
    from mobile.gen.graph import build_graph
    from mobile.gen.ops import load_runnable_opnodes
    ops = load_runnable_opnodes()
    sched_order = list(range(len(ops)))
    random.Random(0xC0FFEE).shuffle(sched_order)
    base = zlib.crc32(producer_id.encode()) & 0xFFFFFFFF
    sched_offset = (base + seed) % len(ops)
    rng = random.Random(seed * 1_000_003 + (base ^ cur) * 2_654_435_761 + 1)
    seed_op = ops[sched_order[(sched_offset + cur) % len(ops)]]
    graph = None
    for _ in range(8):
        graph = build_graph(rng, ops, nodes, leaf_prob, seed_op=seed_op,
                            out_alias_prob=out_alias_prob)
        if graph is not None:
            break
    if graph is None:
        return None, None
    try:
        src = graph.emit(seed=seed)
    except Exception:
        src = None
    return graph, src


def run_pregen(out: str, count: int, nodes: int, leaf_prob: float, seed: int,
               producer_id: str, backend: str = "portable", quantize: bool = False,
               out_alias_prob: float = 0.1) -> int:
    # Lazy probe of ONLY the target backend — never load the other twelve (cost, and it keeps
    # a hostile SDK like QNN out of an openvino worker). available_backends() (full sweep) is
    # paid only on the error path, to list the alternatives.
    if get_backend(backend) is None:
        print(f"[pregen {producer_id}] unknown backend {backend!r}; "
              f"available here: {', '.join(available_backends())}", flush=True)
        return 1
    print(f"[pregen {producer_id}] loading op set (backend={backend}) ...", flush=True)
    ops = load_runnable_opnodes()
    # crc32(id) — stable across processes/runs so a job_id reproduces its exact graph.
    base = zlib.crc32(producer_id.encode()) & 0xFFFFFFFF

    # Deterministic round-robin seed schedule: instead of uniform-then-feasibility-
    # filtered seed picks (which structurally starve constraint-heavy ops), every op
    # occupies the guaranteed seed slot an equal number of times. The op *order* is a
    # fixed shuffle shared by all workers (so the global schedule is well-defined and
    # reproducible); each worker is phase-shifted by its own offset so the fleet covers
    # different ops at the same instant. Pure function of `cur` ⇒ resume/skip-safe.
    sched_order = list(range(len(ops)))
    random.Random(0xC0FFEE).shuffle(sched_order)
    sched_offset = (base + seed) % len(ops)

    crash_dir = Path(out) / "_crashes"
    crash_dir.mkdir(parents=True, exist_ok=True)
    marker = crash_dir / f"{producer_id}.py"        # the graph currently being lowered

    # Indices the fleet recorded as crashing this worker — skip them on respawn.
    skip_file = crash_dir / f"{producer_id}.skip"
    skipped = set()
    if skip_file.exists():
        skipped = {int(x) for x in skip_file.read_text().split() if x.strip().lstrip('-').isdigit()}
    # Resume: indices already written (don't redo the slice after a respawn).
    done_idx = set()
    for p in Path(out, producer_id).glob(f"{producer_id}_*.job"):   # this worker's shard
        try:
            done_idx.add(int(p.stem.rsplit("_", 1)[1]))
        except (ValueError, IndexError):
            pass
    written = len(done_idx)
    if skipped or done_idx:
        print(f"[pregen {producer_id}] resume: {len(done_idx)} already written, "
              f"skipping {len(skipped)} crashed", flush=True)

    # ---- run stats (for the fleet's end-of-run report) --------------------------------
    # Per-INDEX outcome map: cur -> the pipeline stage that index ended at
    # (ready | gen_fail | emit | eager | export | transformation | lower | quant_ref |
    # build_raise). Keyed by index, NOT a running counter, because a respawn re-processes
    # every not-yet-written, non-crashed index — a running counter + reload would
    # TRIPLE-count those failures on each respawn (that is why the report once showed
    # gen_fail ≈ 20% and READY far below the real job count). Overwriting outcomes[cur] is
    # idempotent (the outcome is a deterministic function of cur), so counts derived from
    # this map are respawn-proof. Crashes are owned by the FLEET (it can't be recorded here
    # — the process dies mid-build_job), so `skipped` indices are intentionally absent.
    stats_dir = Path(out) / "_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_file = stats_dir / f"{producer_id}.json"
    outcomes: dict[str, str] = {}
    build_time_sum = 0.0      # seconds spent inside build_job (eager+export+lower)
    build_count = 0           # number of build_job calls timed
    gen_time_sum = 0.0        # seconds spent in graph build + emit
    gen_count = 0
    if stats_file.exists():
        try:
            prev = json.loads(stats_file.read_text())
            outcomes = dict(prev.get("outcomes", {}))
            build_time_sum = float(prev.get("build_time_sum", 0.0))
            build_count = int(prev.get("build_count", 0))
            gen_time_sum = float(prev.get("gen_time_sum", 0.0))
            gen_count = int(prev.get("gen_count", 0))
        except (ValueError, OSError):
            pass  # corrupt/partial file → start this worker's tally fresh
    # Already-written jobs (this shard's .job files on disk) are READY by definition. Seed
    # them so a resume counts them even though the loop `continue`s straight past them.
    for _c in done_idx:
        outcomes[str(_c)] = "ready"

    # ---- failure DETAIL log (so a non-ready stage is diagnosable without replaying) ---
    # stage_counts records only HOW MANY attempts fell out at each stage; the *reason*
    # (the ExportResult.detail string, the emit exception) was discarded, so diagnosing
    # e.g. the emit bucket meant regenerating the whole corpus. Instead write ONE FILE PER
    # FAILED GRAPH — _fails/<producer_id>_<cur>.py — mirroring the corpus's own per-job .py
    # and the _crashes/<id>_<n>.py convention: a reason header (stage / job_id / seed_op /
    # detail) followed by the emitted source (or the graph description when the graph was
    # un-emittable). Written the instant the failure happens, so a later hard-crash can't
    # lose it; the filename keys on cur, so a resume re-attempting an index just overwrites.
    fails_dir = Path(out) / "_fails"
    fails_dir.mkdir(parents=True, exist_ok=True)

    def _record_fail(stage: str, cur: int, seed_op: str, detail: str,
                     graph=None, src: str | None = None) -> None:
        header = (f"# stage:   {stage}\n"
                  f"# job_id:  {producer_id}:{cur}\n"
                  f"# seed_op: {seed_op}\n"
                  f"# reason:  {detail}\n")
        if src is not None:
            body = src                                       # the emitted, runnable source
        elif graph is not None:
            body = f"# (un-emittable — graph description only)\n# {graph.describe()}\n"
        else:
            body = ""
        try:
            (fails_dir / f"{producer_id}_{cur}.py").write_text(header + body)
        except OSError:
            pass                                             # a failed diag write never sinks a run

    def _flush_stats() -> None:
        # `stages` is derived from the per-index map, so it is the same no matter how many
        # respawns produced `outcomes`. `outcomes` is persisted so the next respawn resumes
        # the map (not a lossy count).
        tmp = stats_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "producer_id": producer_id,
            "stages": dict(Counter(outcomes.values())),
            "outcomes": outcomes,
            "build_time_sum": build_time_sum,
            "build_count": build_count,
            "gen_time_sum": gen_time_sum,
            "gen_count": gen_count,
        }))
        tmp.replace(stats_file)  # atomic — the fleet may read this concurrently

    # A graceful stop (Ctrl-C / fleet shutdown) is NOT a crash — persist stats, clear the
    # marker, and go. Flushing here means an interrupted run still reports what it did.
    def _graceful(*_):
        try:
            _flush_stats()
        except Exception:
            pass
        marker.unlink(missing_ok=True)
        os._exit(0)
    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    idx = 0
    while idx < count:                              # count = number of graph ATTEMPTS, not
        cur = idx                                   # successes — crashes/fails consume a slot
        idx += 1                                    # and we move on. lower-rate = jobs / count.
        if cur in skipped or cur in done_idx:
            continue                                # known crash / already written → skip
        rng = random.Random(seed * 1_000_003 + (base ^ cur) * 2_654_435_761 + 1)
        # This index's scheduled seed op (kept fixed across the 8 retries so the
        # coverage intent holds; generate() is stochastic, so retries can still
        # succeed for a seed-finicky op).
        seed_op = ops[sched_order[(sched_offset + cur) % len(ops)]]
        graph = None
        gen_t0 = time.perf_counter()
        for _ in range(8):
            graph = build_graph(rng, ops, nodes, leaf_prob, seed_op=seed_op,
                                out_alias_prob=out_alias_prob)
            if graph is not None:
                break
        if graph is None:
            outcomes[str(cur)] = "gen_fail"         # generator couldn't build a valid DAG
            continue
        job_id = f"{producer_id}:{cur}"
        try:
            src = graph.emit(seed=seed)
        except Exception as e:
            outcomes[str(cur)] = "emit"
            _record_fail("emit", cur, seed_op.op_name,
                         f"emit raised {type(e).__name__}: {e}", graph=graph)
            continue
        if src is None:
            outcomes[str(cur)] = "emit"             # un-emittable → skip
            _record_fail("emit", cur, seed_op.op_name,
                         "emit returned None (un-emittable arg / no schema)", graph=graph)
            continue
        gen_dt = time.perf_counter() - gen_t0

        # Heartbeat for the fleet's hang-watchdog: a worker stuck inside build_job emits
        # no further line, so the fleet (reading our stdout) detects the stall. The index
        # rides along so the fleet knows what to skip.
        print(f"__HB__ {cur}", flush=True)
        marker.write_text(f"# crashed in build_job (eager run or lowering) — job_id {job_id}\n{src}")
        build_t0 = time.perf_counter()
        try:
            job = build_job(src, backend, quantize)  # eager run + lowering; may HARD-CRASH
        except Exception as e:
            outcomes[str(cur)] = "build_raise"      # catchable error → SKIP (marker reused)
            _record_fail("build_raise", cur, seed_op.op_name,
                         f"{type(e).__name__}: {e}", src=src)
            continue
        build_dt = time.perf_counter() - build_t0
        outcomes[str(cur)] = job.stage              # bucket by where it ended (ready = success)
        if job.status != "READY":
            _record_fail(job.stage, cur, seed_op.op_name, job.detail, src=src)
            continue                                # eager-raise / export/lower-fail → skip
        # Successful build only: record timing so the average reflects real job cost, not
        # cheap early failures (per user's request).
        gen_time_sum += gen_dt
        gen_count += 1
        build_time_sum += build_dt
        build_count += 1
        delegated = {"ops": job.delegated_ops, "non": job.non_delegated_ops,
                     "calls": job.delegate_calls}
        frames = P.encode_pushjob(job_id, job.pte, job.inputs, job.eager,
                                  desc=graph.describe(), user_pos=job.user_pos,
                                  delegated=delegated)
        # Annotate the inspectable source with the delegation breakdown (backend=how many
        # ops the partitioner absorbed vs left on portable) so single-op corpus is greppable.
        src_annotated = (f"# delegated: backend={backend} ops={job.delegated_ops} "
                         f"non_delegated={job.non_delegated_ops} "
                         f"delegate_calls={job.delegate_calls}\n{src}")
        C.write_job(out, job_id, frames, src=src_annotated)
        written += 1
        if written % 100 == 0:
            print(f"[pregen {producer_id}] {written}/{count} → {out}", flush=True)
            _flush_stats()                          # periodic checkpoint (survives a crash)

    marker.unlink(missing_ok=True)                  # finished cleanly → no crash
    _flush_stats()
    avg_ms = (build_time_sum / build_count * 1e3) if build_count else 0.0
    print(f"[pregen {producer_id}] done: {written} jobs in {out}/  "
          f"(avg build {avg_ms:.0f} ms over {build_count} READY)", flush=True)
    return 0
