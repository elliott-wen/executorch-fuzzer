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

import random
import signal
import zlib
from pathlib import Path

from mobile.gen.graph import build_graph
from mobile.gen.ops import load_runnable_opnodes
from mobile.gen.export import build_job, get_backend, available_backends
from mobile.net import protocol as P
from mobile.net import corpus as C


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

    # A graceful stop (Ctrl-C / fleet shutdown) is NOT a crash — clear the marker and go.
    def _graceful(*_):
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
        for _ in range(8):
            graph = build_graph(rng, ops, nodes, leaf_prob, seed_op=seed_op,
                                out_alias_prob=out_alias_prob)
            if graph is not None:
                break
        if graph is None:
            continue
        job_id = f"{producer_id}:{cur}"
        try:
            src = graph.emit(seed=seed)
        except Exception:
            continue
        if src is None:
            continue                                # un-emittable → skip

        # Heartbeat for the fleet's hang-watchdog: a worker stuck inside build_job emits
        # no further line, so the fleet (reading our stdout) detects the stall. The index
        # rides along so the fleet knows what to skip.
        print(f"__HB__ {cur}", flush=True)
        marker.write_text(f"# crashed in build_job (eager run or lowering) — job_id {job_id}\n{src}")
        try:
            job = build_job(src, backend, quantize)  # eager run + lowering; may HARD-CRASH
        except Exception:
            continue                                # catchable error → SKIP (marker reused)
        if job.status != "READY":
            continue                                # eager-raise / export-fail → skip
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

    marker.unlink(missing_ok=True)                  # finished cleanly → no crash
    print(f"[pregen {producer_id}] done: {written} jobs in {out}/", flush=True)
    return 0
