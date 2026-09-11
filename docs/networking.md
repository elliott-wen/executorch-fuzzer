# 6. Networking (broker / feed / client)

The online half replays a corpus across a fleet of executors over **ZeroMQ**. Three
roles around one broker; the broker is a pure router, the feeder owns the diff, the
worker just runs. This is what lets executors live on other machines and (later) phones.

## Roles

| role | code | socket | responsibility |
|------|------|--------|----------------|
| **broker** | `executor/broker.py` | 2× ROUTER + PUB | route job/result frames; never touch a tensor |
| **feed** | `executor/feed.py` | DEALER ↔ frontend | read corpus, send lean jobs, **diff** results vs eager, tally, log |
| **client** | `net/client.py` | REQ ↔ backend | LRU work-pull; run the `.pte`, return raw outputs |

```
feed (DEALER) ──jobs──►  broker frontend ROUTER
                          │  job_id → feeder map
                          │  LRU queue of free workers
feed ◄──results────────  broker, routed back by job_id
                          backend ROUTER ◄──REQ──► client ×K
                          ctrl PUB ──STOP──► clients (clean shutdown)
```

## The protocol (`executor/protocol.py`)

Language-neutral, pickle-free, binary — so a native phone client can speak it. Every
message is a list of byte frames: frame 0 is a small JSON header, the rest are raw
binary (the `.pte` and tensor buffers). Tensors serialize as `{dtype, dims}` in the
header plus a contiguous little-endian raw buffer. `dtype` is the model's ScalarType
integer code, so the phone, the broker, and the Z3 model all agree on numbering.

Three message shapes:

| message | direction | payload |
|---------|-----------|---------|
| `pushjob` | producer → broker | header{job_id, inputs[], eager[]} + pte + input-raws + eager-raws |
| `job` | broker → client | header{job_id, inputs[]} + pte + input-raws |
| `result` | client → broker | header{job_id, status, detail, outputs[]} + output-raws |

The broker routes by reading the cleartext `job_id` from frame 0 only — it never
decompresses a payload frame.

## The broker is a pure switch

Two bound ROUTERs (feeders on the jobs port, workers on the clients port) plus a PUB
for STOP. State is minimal and routing-only: an LRU queue of free workers and a
`job_id → feeder` map. Because the feeder holds the eager reference and does the diff
itself, the broker only has to get each worker's result back to the feeder that sent
the job. Scale feeders and workers freely; the broker stays trivial.

## The feeder owns the diff (`executor/feed.py`)

The feeder is the single entry point for both bulk fuzzing and single-graph debugging:

```bash
mobile feed --corpus corpus/portable                 # whole corpus → TSV + tally
mobile feed w33:531 --corpus corpus/portable          # one job_id → rich JSON observation
mobile feed path/to/w33_531.job                       # one graph by .job path
mobile feed --from-tsv tmp/skip.tsv --status MISMATCH --limit 20   # replay failing rows
```

It reads the corpus (so it holds the eager reference), sends each lean job (pte +
inputs), receives the worker's raw outputs, diffs them against its kept eager
(`executor/compare.py`), tallies, and writes the skip-log. A bounded in-flight **window**
gives end-to-end backpressure; a job with no result inside the timeout is recorded
**TIMEOUT**.

Single-graph / `--from-tsv` runs print one JSON object per job (job_id, status, detail,
op_chain, per-output `{dtype, shape, max_abs_delta, eager, et}`) — the debug
observation. *Where* it ran is just which worker happened to be connected (a local
`mobile client` on x86, or a phone on ARM).

## The client is a thin executor (`net/client.py`)

One client = one worker = one REQ connection. Thin by design: ExecuTorch runtime +
tensor I/O only — no z3, no export, no generation. It is crash-isolated like the phone
will be: a stable **coordinator** (holds the broker connection) drives a disposable
**warm executor child** (forked, holds the runtime).

```
send READY → recv JOB(pte+inputs) → run → send RESULT(raw outputs) → recv JOB → ...
(a SUB channel delivers STOP for clean shutdown)
```

A native abort kills only the child → coordinator reports **CRASH** and respawns; a
hung kernel is killed at `--job-timeout` → **TIMEOUT** + respawn. The worker *always*
returns a verdict, so nothing relies on the broker detecting a disconnect.

## Supervisors

- **`local_fleet.py`** — spawns and auto-respawns the feeder + a fleet of clients:
  ```bash
  python mobile/local_fleet.py --clients 16 --corpus corpus/portable --host 127.0.0.1
  ```
  Omit `--corpus` to manage only clients and run `mobile feed` yourself.
- **`spawn_clients.sh`** — shell helper to spin up bare clients.

## Across machines / phones

Run the broker on one host; point `feed` and `client` at it with `--host <broker-ip>`
and open ports **15554** (jobs), **15555** (clients), **15556** (ctrl). Scale execution
by adding clients on more machines or devices — the phone is a future `client` speaking
the same binary protocol.
