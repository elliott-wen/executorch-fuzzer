# 5. Corpus & pre-generation

Generation is **offline and pre-generated to disk**: graphs are built and lowered
once, into a corpus, then replayed by the online fleet. This makes runs reproducible
and every graph inspectable. Code: `executor/pregen.py` (one worker), `pregen_fleet.py`
(the parallel supervisor), `executor/corpus.py` (the on-disk store), `run_pregen.sh`
(turnkey launcher).

## What a corpus looks like

```
corpus/<out>/
  <id>/<id>_<idx>.job      # the wire payload: pte + inputs + eager + desc + user_pos
  <id>_<idx>.py            # the SAME graph as a standalone runnable script (the repro)
  _crashes/
    <id>.py                # the in-flight graph (marker), overwritten before each lower
    <id>.skip              # indices this worker must skip (crashed/hung there)
    <id>_<idx>.py          # preserved crash/hang samples
```

A broker row `MISMATCH p0:12` ⇒ open `corpus/<out>/p0_12.py`.

## One worker: `run_pregen`

For each index `cur` in its slice (`executor/pregen.py`):

1. seed the graph with this index's **scheduled** op (round-robin — see
   [graph-generation.md](graph-generation.md)),
2. `build_graph` (up to 8 retries; `None` ⇒ skip the index),
3. `emit()` → source (`None` ⇒ skip),
4. **emit a heartbeat** `__HB__ <cur>` to stdout (for the fleet watchdog),
5. write the crash **marker** (the in-flight graph), then `build_job` — the risky
   eager-run + lowering that may **hard-crash or hang**,
6. on `READY`, write the `.job` + `.py` to the corpus; else skip.

### Determinism & resume

- The RNG is `random.Random(seed * 1_000_003 + (base ^ cur) * 2_654_435_761 + 1)`
  where `base = crc32(producer_id)` — so a `(seed, id, cur)` triple always reproduces
  the exact graph, across processes and runs.
- On restart a worker **resumes**: it scans its already-written `.job` files and the
  `.skip` list and skips those indices, continuing past prior work and known-bad graphs.

## The fleet: `pregen_fleet.py`

Runs N workers in parallel, each generating a **disjoint** slice (distinct `--id` ⇒
crc32-disjoint graph space + distinct filenames) into one `--out`. It staggers
launches, supervises, and stops when all workers finish.

```bash
python mobile/pregen_fleet.py --workers 96 --total 100000 --out corpus/portable --backend portable
```

`--total` splits a target across workers; `--per-worker` sets a per-worker count.

## Crash and hang recovery

Running an arbitrary graph through eager + lowering is unsafe. Three failure modes,
all handled:

| failure in `build_job` | process | recovery |
|------------------------|---------|----------|
| catchable exception | raises | worker catches → **SKIP**, continues |
| hard crash (segfault/abort) | **dies** (rc≠0) | fleet reads the marker, records the index in `.skip`, preserves the sample, **respawns** |
| **hang** (infinite/pathological-slow eager/export/lowering) | **alive, no progress** | hang watchdog (below) kills + skips + respawns |

### The hang watchdog (stdout heartbeat)

A hanging graph never crashes, so the marker/`.skip`/respawn path keyed on process
death never fires — without protection the fleet would wait on a stuck worker forever
(the classic "stalls at ~9700/10000 and never finishes").

The fix is a **stdout heartbeat**, FS-independent:

- each worker prints `__HB__ <idx>` right before every `build_job`,
- the fleet reads each worker's stdout over a pipe (one reader thread per worker),
  updates `last_beat[i] = monotonic()` on **every** line, and tees real output to the
  worker's log,
- the monitor loop kills any *alive* worker idle longer than `--worker-timeout`
  (default **300s**), then routes it through the same skip+respawn path as a crash.

A worker stuck *inside* `build_job` emits no further line, so the idle gap is exactly
the hang. Liveness is a monotonic clock over a kernel pipe — no filesystem mtime, no
clock skew, works identically on local or shared storage. `RESPAWN_CAP` (200) bounds
runaway respawning.

> Earlier this used the crash-marker file's mtime as the heartbeat; that worked on a
> local FS but coupled liveness to filesystem semantics. The stdout heartbeat replaces
> it with a direct parent↔child signal.

## Turnkey launcher: `run_pregen.sh`

```bash
mobile/run_pregen.sh [BACKEND] [WORKERS] [TOTAL] [OUT]
mobile/run_pregen.sh qualcomm 24 100000          # backend, workers, total
NODES=4 mobile/run_pregen.sh xnnpack 32 50000     # override node count via env
```

It guards against launching a second fleet while one is running.

## Monitoring & stopping a run

```bash
# progress
find corpus/<out> -name '*.job' | wc -l

# the fleet prints, every 15s:
#   workers done X/N  |  corpus M jobs
# and per recovery:
#   wI hung 312s (no stdout) — killing
#   wI crashed on idx 5123 — saved + respawn (skipping it) [2]

# stop: SIGTERM the fleet parent; it propagates to workers (graceful, not a crash)
kill <fleet-pid>
```

A graceful stop clears the in-flight marker (it is *not* recorded as a crash), so a
later resume doesn't skip that index.
