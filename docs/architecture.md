# 2. Architecture

The pipeline has two halves that are deliberately decoupled: **offline generation**
(CPU-heavy, parallel, reproducible) and **online execution** (a ZeroMQ-brokered
fleet of executors that can live on other machines/phones).

## End-to-end data flow

```
 OFFLINE (once)                          ONLINE (replay, scalable)
 ┌─────────────────────┐
 │ pregen / fleet      │   corpus/       ┌──────────────────────────────────────────┐
 │  generate graph     │──►  .job  ──────►│ feed                                       │
 │  run eager (oracle) │   .py (repro)    │  read corpus (holds eager ref)             │
 │  lower → .pte       │                  │  PUSH lean job (pte+inputs) ─► broker       │
 └─────────────────────┘                  │  diff worker output vs eager ◄─ broker      │
                                          │  tally + skip-log + samples                 │
                                          └───────────────┬────────────────────────────┘
                                                          │ ZeroMQ
                                            ┌─────────────▼─────────────┐
                                            │ broker (pure router)      │
                                            │  feeders ↔ workers (LRU)  │
                                            └─────────────┬─────────────┘
                                                          │
                                       ┌──────────────────┼──────────────────┐
                                       ▼                  ▼                  ▼
                                  client ×K          client          client (phone, future)
                                  run .pte, return raw outputs (no diff, no generation)
```

## Offline: generation → corpus

A `pregen` worker, for each index in its slice:

1. builds a valid graph (`gen/graph/build.build_graph`),
2. emits it as standalone source (`gen/graph/ir.GenGraph.emit`),
3. runs the eager reference and lowers to `.pte` (`gen/export/job.build_job`),
4. writes a `.job` (the wire payload) + a `.py` (the human-readable repro) to the corpus.

Generation is parallelized by `pregen_fleet.py` across N workers writing **disjoint**
slices into one corpus dir. See [corpus-and-pregen.md](corpus-and-pregen.md) and
[graph-generation.md](graph-generation.md).

A `.job` carries everything the executor needs and the feeder needs to diff:
`pte`, the concrete `inputs`, the `eager` reference outputs, a `desc` (op-chain), and
`user_pos` (which output positions are real graph outputs vs mutated-input aliases).

## Online: the ZeroMQ topology

Three roles, one broker:

| role | socket | does |
|------|--------|------|
| **broker** (`executor/broker.py`) | two ROUTERs + a PUB | pure switch: routes job frames feeder→worker and result frames worker→feeder; never decompresses a payload |
| **feed** (`executor/feed.py`) | DEALER ↔ broker frontend | reads corpus, sends lean jobs, receives raw results, **does the diff** vs its kept eager, tallies, writes skip-log + samples |
| **client** (`net/client.py`) | REQ ↔ broker backend | LRU work-pull: `READY → JOB → run .pte → RESULT`; runs the program, returns raw outputs |

Key property: **the broker never touches a tensor.** It routes by peeking the
cleartext `job_id` in each frame's small JSON header (the payload frames — `.pte`,
tensor buffers — are never decompressed). State is minimal: an LRU queue of free
workers and a `job_id → feeder` map. Scale feeders (each diffs its own jobs) and
workers independently; the broker is just the switch.

Backpressure: the feeder keeps a bounded in-flight **window**, so it blocks when
executors lag — end-to-end flow control without the broker buffering work.

## Why the diff lives in the feeder, not the broker or worker

The feeder already read the corpus, so it already holds the eager reference. Putting
the comparison there means:

- the **broker** stays a trivial, stateless router (easy to trust and scale),
- the **worker** stays thin — it runs a `.pte` and returns bytes, nothing else. That
  is exactly what a phone can do, in any language, with no z3/export/torch-Python.

The worker speaks a **language-neutral binary protocol** (`executor/protocol.py`): every
message is a list of byte frames — a small JSON header plus raw little-endian tensor
buffers and the `.pte`. No pickle, no numpy — a native C++/Kotlin phone client can
speak it directly.

## Crash isolation

Running an arbitrary generated graph is unsafe: a kernel can segfault or hang. The
executor `client` is structured like the phone will be:

- a stable **coordinator** (holds the broker connection),
- driving a disposable **warm executor child** (a forked process holding the runtime).

A native abort kills only the child → the coordinator reports **CRASH** and respawns
it. A hung kernel is killed at the deadline → **TIMEOUT** + respawn. So the worker
*always* returns a verdict; nothing relies on the broker noticing a disconnect. The
generation side has its own crash/hang protection — see
[corpus-and-pregen.md](corpus-and-pregen.md).

## Where things run

- **Generation** runs on the build host (x86, CPU). Heavy: torch + executorch.exir +
  the full op set per worker.
- **Execution** runs wherever a `client` connects — the same host, another machine, or
  (future) an ARM phone. Point `feed`/`client` at the broker with `--host <broker-ip>`
  and open ports 15554–15556.
