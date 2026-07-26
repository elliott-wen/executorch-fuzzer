#!/usr/bin/env python3
"""pregen_fleet.py — run N pregen workers in parallel to build a corpus fast.

Each worker generates a DISJOINT slice of the corpus (distinct --id ⇒ crc32-disjoint
graph space + distinct filenames) into the same --out dir, then exits. This is a BATCH
job: the script staggers the launches, waits for every worker to finish, and stops when
all are done. A worker that hard-crashes OR hangs (no stdout for --worker-timeout) is
killed, its offending graph saved to <out>/_crashes/w<i>_<idx>.py, the index recorded in
its .skip list, and the worker RESPAWNED to skip past it (up to RESPAWN_CAP times).
Liveness is read from each worker's stdout — no filesystem/mtime dependence.

  python mobile/pregen_fleet.py --workers 64 --per-worker 200 --out tmp/corpus
  python mobile/pregen_fleet.py --workers 64 --total 100000 --out tmp/corpus   # split a target

Total jobs ≈ workers × per-worker (or --total split across workers).

NOTE: each worker loads torch + executorch.exir + the full op set (~1-2 GB, ~75s) and does
heavy to_executorch lowering. 128 in parallel is a lot of RAM/CPU — size --workers to the
box (≈ nproc, watch `free -g`).
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# The ONLY thing the parent dir is used for: putting it on PYTHONPATH so child workers
# can `import mobile`. Nothing is ever written here — outputs go to the launch cwd.
IMPORT_ROOT = Path(__file__).resolve().parent.parent
# Workers run the repo-root pregen.py launcher directly (it self-injects sys.path), so no
# `-m mobile` is needed. Lives next to this file.
PREGEN = str(Path(__file__).resolve().parent / "pregen.py")
# Spawn workers with the same interpreter that launched this fleet (the co-located
# .venv), so it survives the venv being moved. Its bin dir goes on PATH because
# executorch lowering shells out to `flatc` (FlatBuffers compiler), which lives there.
PY = sys.executable
_BIN = str(Path(PY).resolve().parent)
# CPU-only by default: every mobile-NPU backend lowers on CPU, so workers blank the GPU.
# EXCEPTION: the CUDA/AOTInductor backend COMPILES its kernels on a real GPU at lower time,
# so if the launcher exported a non-empty CUDA_VISIBLE_DEVICES we inherit it (and, below,
# round-robin the workers across those GPUs). Unset → "" preserves the old behavior exactly.
_GPUS = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
ENV = {**os.environ, "PYTHONPATH": str(IMPORT_ROOT),
       "CUDA_VISIBLE_DEVICES": ",".join(_GPUS),
       "PATH": _BIN + os.pathsep + os.environ.get("PATH", "")}
RESPAWN_CAP = 200   # safety net: each respawn skips one crash, so this only trips if a
                    # worker keeps crashing on different graphs (or OOMs) endlessly


def main() -> int:
    ap = argparse.ArgumentParser(description="parallel pre-generation fleet")
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--per-worker", type=int, default=100, help="jobs each worker generates")
    ap.add_argument("--total", type=int, default=0,
                    help="if set, split this many jobs across workers (overrides --per-worker)")
    ap.add_argument("--out", default="tmp/corpus")
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--leaf-prob", type=float, default=0.3)
    ap.add_argument("--out-alias-prob", type=float, default=0.1,
                    help="chance a grow node's out= buffer aliases a live producer "
                         "(exercises compiler memory-planning) vs. a fresh allocation")
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--quantize", action="store_true",
                    help="PT2E-quantize before lowering (backends that support it)")
    ap.add_argument("--backend", default="portable",
                    help="lowering target for every worker: portable | xnnpack | vulkan | ...")
    ap.add_argument("--stagger", type=float, default=0.3, help="seconds between launches")
    ap.add_argument("--concurrency", type=int, default=0,
                    help="max workers RUNNING AT ONCE (0 = all, the default = full parallelism). "
                         "The remaining workers queue and start in id order as slots free up. Use "
                         "for GPU-bound backends whose per-graph lowering runs ON the device: a "
                         "CUDA fleet with N-per-GPU thrashes the autotuner, so cap it (e.g. "
                         "--concurrency 7 = one worker per GPU, or --concurrency 1 = strictly w0's "
                         "whole slice, then w1, then w2 ...). Sharding/seeds are UNCHANGED, so the "
                         "corpus is byte-identical to a full-parallel run — only the schedule differs.")
    ap.add_argument("--worker-timeout", type=float, default=300.0,
                    help="kill+skip+respawn a worker that has emitted no stdout for this "
                         "many seconds (a HANG in eager/export/lowering that never crashes "
                         "— otherwise the fleet waits on it forever). Liveness is the "
                         "worker's stdout stream (no filesystem/clock dependence).")
    ap.add_argument("--logdir", default="tmp/pregen_logs")
    ap.add_argument("--cache-dir", default="",
                    help="base dir for PER-WORKER torch-inductor / triton COMPILE caches "
                         "(default: <out>/_compile_cache). CRITICAL for the CUDA/AOTInductor "
                         "backend: workers each compile triton kernels, and the default shared "
                         "cache (/tmp/torchinductor_<user>) RACES across concurrent workers — one "
                         "writing a half-built .so while another mmaps it fails with 'failed to "
                         "map segment from shared object', which surfaces as flaky "
                         "'Failed to run autotuning code block' lower failures across every op. "
                         "/tmp is also small + often noexec. Per-worker dirs on the out disk fix "
                         "both. Harmless for CPU backends that never touch inductor.")
    a = ap.parse_args()

    per_worker = (a.total + a.workers - 1) // a.workers if a.total else a.per_worker
    # Resolve --out/--logdir to ABSOLUTE paths against the fleet's launch cwd before they
    # reach the workers. Each worker is a subprocess whose cwd is not guaranteed to match
    # ours (e.g. under `screen`, whose window can start in $HOME), so a relative --out
    # would have the fleet and the workers write to / glob two different dirs — the corpus
    # would only "show up" with an absolute path. Pin it here so everyone agrees.
    a.out = str(Path(a.out).resolve())
    a.logdir = str(Path(a.logdir).resolve())
    # Per-worker compile-cache base, on the (roomy, exec-allowed) out disk by default — never
    # the shared /tmp inductor cache that races across concurrent CUDA workers. See --cache-dir.
    cache_base = Path(a.cache_dir).resolve() if a.cache_dir else Path(a.out) / "_compile_cache"
    cache_base.mkdir(parents=True, exist_ok=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    logdir = Path(a.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    def cmd(i: int) -> list[str]:
        return [PY, PREGEN, "--out", a.out, "--id", f"w{i}",
                "--count", str(per_worker), "--seed", str(a.seed),
                "--nodes", str(a.nodes), "--leaf-prob", str(a.leaf_prob),
                "--out-alias-prob", str(a.out_alias_prob),
                "--backend", a.backend] + (["--quantize"] if a.quantize else [])

    procs: dict[int, tuple[subprocess.Popen, threading.Thread]] = {}
    done: set[int] = set()
    respawns: dict[int, int] = {}
    events = {"crashed": 0, "hung": 0}     # hard process deaths, by kind (fleet-observed —
                                           # the worker can't record its own crash/hang stage)
    last_beat: dict[int, float] = {}       # monotonic time of each worker's last stdout line
    last_idx: dict[int, str] = {}          # in-flight index from the worker's last __HB__
    stop = {"flag": False}

    def on_sig(_s, _f):
        stop["flag"] = True
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    def reader(i: int, proc: subprocess.Popen, fh) -> None:
        """Drain a worker's stdout: every line is a liveness beat (updates last_beat);
        __HB__ lines carry the in-flight index; all other output is teed to its log."""
        try:
            for raw in proc.stdout:                # blocks until a line or EOF (worker exit)
                last_beat[i] = time.monotonic()
                if raw.startswith(b"__HB__"):
                    parts = raw.split()
                    if len(parts) >= 2:
                        last_idx[i] = parts[1].decode("ascii", "ignore")
                else:
                    fh.write(raw)                  # tee real output to the log (binary)
                    fh.flush()
        finally:
            fh.close()

    def launch(i: int) -> None:
        fh = open(logdir / f"w{i}.log", "ab")
        # --out/--logdir were made absolute in main(), so the worker's cwd is irrelevant to
        # where the corpus lands (it need not match ours under `screen` et al.). Imports
        # work via PYTHONPATH=IMPORT_ROOT in ENV.
        env = ENV
        if _GPUS:                                   # CUDA lane: pin each worker to one GPU
            env = {**ENV, "CUDA_VISIBLE_DEVICES": _GPUS[i % len(_GPUS)]}  # cuda:0 == this GPU
        # Isolate this worker's triton/inductor compile cache so concurrent workers never race
        # the same on-disk .so (the corpus_v5/cuda 'failed to map segment' / autotuning-subprocess
        # flakiness). Keyed by worker id so a respawn reuses its own warm cache. Non-CUDA workers
        # set these harmlessly. `torchinductor`/`triton` subdirs mirror the default layout.
        wcache = cache_base / f"w{i}"
        (wcache / "inductor").mkdir(parents=True, exist_ok=True)
        (wcache / "triton").mkdir(parents=True, exist_ok=True)
        env = {**env,
               "TORCHINDUCTOR_CACHE_DIR": str(wcache / "inductor"),
               "TRITON_CACHE_DIR": str(wcache / "triton")}
        p = subprocess.Popen(cmd(i), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=env)
        last_beat[i] = time.monotonic()           # grace window starts at launch
        t = threading.Thread(target=reader, args=(i, p, fh), daemon=True)
        t.start()
        procs[i] = (p, t)

    conc = a.concurrency if a.concurrency > 0 else a.workers   # 0 → all at once (old behavior)
    started = {"n": 0}                        # next worker id not yet launched (started in order)

    def launch_next() -> None:
        """Start the next queued worker(s) into any free slot, up to `conc` running at once. With
        conc==workers this launches everyone up front; with conc==1 it starts w0 and only starts
        w1 after w0's whole slice finishes — strict one-at-a-time. recover() respawns re-occupy a
        worker's own slot, so they don't consume queue headroom."""
        while not stop["flag"] and started["n"] < a.workers and len(procs) < conc:
            launch(started["n"])
            started["n"] += 1
            time.sleep(a.stagger)

    print(f"pregen fleet: {a.workers} workers × {per_worker} jobs → {a.out}/  "
          f"(≈{a.workers * per_worker} total, concurrency={conc}, logs {a.logdir}/)", flush=True)
    launch_next()

    def recover(i: int, reason: str) -> None:
        """Shared crash/hang recovery: pull the in-flight index from the worker's marker,
        record it in its .skip list, preserve the sample, then RESPAWN — the worker skips
        that index (and resumes past written jobs) so it goes past the bad graph."""
        events[reason] = events.get(reason, 0) + 1   # tally hard deaths for the final report
        cd = Path(a.out) / "_crashes"
        cidx = None
        marker = cd / f"w{i}.py"
        if marker.exists():
            m = re.search(r"job_id\s+\S+:(-?\d+)", marker.read_text())
            if m:
                cidx = m.group(1)
                with open(cd / f"w{i}.skip", "a") as sf:
                    sf.write(cidx + "\n")
                try:
                    marker.rename(cd / f"w{i}_{cidx}.py")   # keep the sample
                except OSError:
                    pass
        respawns[i] = respawns.get(i, 0) + 1
        if respawns[i] > RESPAWN_CAP:
            print(f"  w{i} {reason} {respawns[i]}× — giving up (see {cd}/)", flush=True)
            done.add(i)
            return
        print(f"  w{i} {reason} on idx {cidx} — saved + respawn (skipping it) "
              f"[{respawns[i]}]", flush=True)
        launch(i)

    last_status = 0.0
    while not stop["flag"] and len(done) < a.workers:
        time.sleep(2.0)
        mono = time.monotonic()
        for i, (p, t) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                # Alive — hang watchdog. A worker stuck in eager/export/lowering emits no
                # more stdout, so its last_beat goes stale while it never crashes; without
                # this the fleet would wait on it forever (the "stuck at ~9700" stall).
                idle = mono - last_beat.get(i, mono)
                if idle <= a.worker_timeout:
                    continue                              # heard from it recently — healthy
                print(f"  w{i} hung {int(idle)}s (no stdout) — killing", flush=True)
                p.kill()
                try:
                    p.wait(timeout=10)
                except Exception:
                    pass
                t.join(timeout=5)                         # let the reader drain EOF + close log
                procs.pop(i)
                recover(i, "hung")
                continue
            t.join(timeout=5)                             # reader finishes on the worker's EOF
            procs.pop(i)
            if rc == 0:                                   # finished its slice
                done.add(i)
                continue
            recover(i, "crashed")                         # crashed mid-slice
        launch_next()                                     # a finished worker frees a slot → start
                                                          # the next queued worker (concurrency cap)
        now = time.time()
        if now - last_status >= 15:
            last_status = now
            njob = sum(1 for _ in Path(a.out).glob("*/*.job"))
            queued = a.workers - started["n"]
            print(f"  running {len(procs)}  done {len(done)}/{a.workers}  queued {queued}  "
                  f"|  corpus {njob} jobs", flush=True)

    if stop["flag"]:
        print("\ninterrupted — terminating workers ...", flush=True)
        for p, t in procs.values():
            p.terminate()
        time.sleep(2)
        for p, t in procs.values():
            if p.poll() is None:
                p.kill()

    njob = sum(1 for _ in Path(a.out).glob("*/*.job"))
    print(f"done: {len(done)}/{a.workers} workers finished — corpus has {njob} jobs in {a.out}/",
          flush=True)
    _report(a.out, events)
    return 0


# The pipeline stages a worker buckets each attempt into, in pipeline order. `ready` is a
# written job; every other stage is WHERE an attempt fell out. `crashed`/`hung` are appended
# by the fleet (a hard death the worker couldn't self-record). See mobile/gen/export/job.py.
_STAGE_ORDER = ["ready", "gen_fail", "emit", "eager", "export", "transformation",
                "lower", "quant_ref", "config", "build_raise", "crashed", "hung"]
_STAGE_DESC = {
    "ready":          "READY — lowered & written",
    "gen_fail":       "graph generation gave up (no valid DAG in 8 tries)",
    "emit":           "source emit/exec failed",
    "eager":          "eager reference raised (invalid-input graph)",
    "export":         "torch.export failed",
    "transformation": "to_edge (ATen→Edge dialect transform) failed",
    "lower":          "to_executorch / delegate lowering failed",
    "quant_ref":      "quantized-reference build failed",
    "config":         "backend/quantize misconfigured",
    "build_raise":    "build_job raised a catchable error",
    "crashed":        "worker process CRASHED in build_job (hard)",
    "hung":           "worker HUNG in build_job (killed by watchdog)",
}


def _report(out: str, events: dict) -> None:
    """Aggregate every worker's _stats/<id>.json + the fleet's own crash/hang tally into a
    single end-of-run summary: yield, average successful-build time, and a per-stage failure
    breakdown (where attempts fell out of the pipeline)."""
    import json
    stages: dict[str, int] = {}
    build_time_sum = 0.0
    build_count = 0
    gen_time_sum = 0.0
    gen_count = 0
    sdir = Path(out) / "_stats"
    for f in sorted(sdir.glob("*.json")) if sdir.exists() else []:
        try:
            d = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        for k, v in d.get("stages", {}).items():
            stages[k] = stages.get(k, 0) + int(v)
        build_time_sum += float(d.get("build_time_sum", 0.0))
        build_count += int(d.get("build_count", 0))
        gen_time_sum += float(d.get("gen_time_sum", 0.0))
        gen_count += int(d.get("gen_count", 0))
    # Fold the fleet-observed hard deaths in (workers can't record their own crash/hang).
    for k in ("crashed", "hung"):
        if events.get(k):
            stages[k] = stages.get(k, 0) + events[k]

    # Every count comes from the per-worker outcome maps, which are now respawn-idempotent
    # (keyed by index, and READY re-seeded from each shard's on-disk .job files at every
    # resume). So the summed stats are self-consistent — no need to re-count the corpus.
    attempts = sum(stages.values())
    ready = stages.get("ready", 0)

    # Collect the report lines so we can BOTH print them and persist them to a file. The
    # timing lines (avg build / avg gen ms/job) are part of this same block, so writing the
    # whole report captures the time measurements too.
    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)
        print(s, flush=True)

    emit("\n" + "=" * 68)
    emit("PREGEN RUN REPORT")
    emit("=" * 68)
    if not attempts:
        emit("  (no per-worker stats found — nothing to report)")
    else:
        emit(f"  attempts        {attempts}")
        emit(f"  READY jobs      {ready}  ({ready / attempts * 100:.1f}% yield)")
        if build_count:
            emit(f"  avg build time  {build_time_sum / build_count * 1e3:.0f} ms/job  "
                 f"(eager+export+lower, successful jobs only, n={build_count})")
        if gen_count:
            emit(f"  avg gen time    {gen_time_sum / gen_count * 1e3:.1f} ms/job  "
                 f"(graph build + emit, successful jobs only)")
        failures = attempts - ready
        emit(f"\n  failure breakdown ({failures} non-READY attempts, by pipeline stage):")
        # Known stages in pipeline order first, then any unexpected keys.
        keys = [k for k in _STAGE_ORDER if k != "ready" and stages.get(k)]
        keys += sorted(k for k in stages if k not in _STAGE_ORDER)
        if failures:
            for k in keys:
                n = stages[k]
                emit(f"    {k:<15} {n:>8}  ({n / attempts * 100:4.1f}% of attempts)  "
                     f"— {_STAGE_DESC.get(k, '?')}")
        else:
            emit("    (none)")
        emit("=" * 68)

    # Persist the same summary next to the corpus so the numbers (incl. the timing lines)
    # survive after the terminal scrollback is gone. Best-effort — never fail the run over it.
    try:
        (Path(out) / "REPORT.txt").write_text("\n".join(lines).lstrip("\n") + "\n")
    except OSError as e:
        print(f"  (could not write REPORT.txt: {e})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
