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
ENV = {**os.environ, "PYTHONPATH": str(IMPORT_ROOT), "CUDA_VISIBLE_DEVICES": "",
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
    ap.add_argument("--worker-timeout", type=float, default=300.0,
                    help="kill+skip+respawn a worker that has emitted no stdout for this "
                         "many seconds (a HANG in eager/export/lowering that never crashes "
                         "— otherwise the fleet waits on it forever). Liveness is the "
                         "worker's stdout stream (no filesystem/clock dependence).")
    ap.add_argument("--logdir", default="tmp/pregen_logs")
    a = ap.parse_args()

    per_worker = (a.total + a.workers - 1) // a.workers if a.total else a.per_worker
    # Resolve --out/--logdir to ABSOLUTE paths against the fleet's launch cwd before they
    # reach the workers. Each worker is a subprocess whose cwd is not guaranteed to match
    # ours (e.g. under `screen`, whose window can start in $HOME), so a relative --out
    # would have the fleet and the workers write to / glob two different dirs — the corpus
    # would only "show up" with an absolute path. Pin it here so everyone agrees.
    a.out = str(Path(a.out).resolve())
    a.logdir = str(Path(a.logdir).resolve())
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
        p = subprocess.Popen(cmd(i), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=ENV)
        last_beat[i] = time.monotonic()           # grace window starts at launch
        t = threading.Thread(target=reader, args=(i, p, fh), daemon=True)
        t.start()
        procs[i] = (p, t)

    print(f"pregen fleet: {a.workers} workers × {per_worker} jobs → {a.out}/  "
          f"(≈{a.workers * per_worker} total, logs {a.logdir}/)", flush=True)
    for i in range(a.workers):
        if stop["flag"]:
            break
        launch(i)
        time.sleep(a.stagger)

    def recover(i: int, reason: str) -> None:
        """Shared crash/hang recovery: pull the in-flight index from the worker's marker,
        record it in its .skip list, preserve the sample, then RESPAWN — the worker skips
        that index (and resumes past written jobs) so it goes past the bad graph."""
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
        now = time.time()
        if now - last_status >= 15:
            last_status = now
            njob = sum(1 for _ in Path(a.out).glob("*/*.job"))
            print(f"  workers done {len(done)}/{a.workers}  |  corpus {njob} jobs", flush=True)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
