# 7. CLI reference

All commands assume:

```bash
cd /data/jwen929/pytorch
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=""
```

`CUDA_VISIBLE_DEVICES=""` is required (the build is CPU-only); the entry points set it
defensively, but export it too.

## `python -m mobile` subcommands

### `pregen` — generate a corpus (one worker)

```bash
python -m mobile pregen --out corpus/portable --count 10000 --id p0
```

| flag | default | meaning |
|------|---------|---------|
| `--out` | `tmp/corpus` | corpus directory to write |
| `--count` | `10000` | READY jobs to write |
| `--nodes` | `16` | target real-op nodes per graph (adapters are extra) |
| `--leaf-prob` | `0.3` | chance a grow port uses a fresh leaf vs. connecting a producer |
| `--out-alias-prob` | `0.1` | chance an `out=` buffer aliases a live producer (memory-planning coverage) |
| `--seed` | `0xC0FFEE` | global seed; with `--id`, fixes every graph |
| `--id` | `p0` | producer id (distinct per parallel worker ⇒ disjoint graph space) |
| `--backend` | `portable` | lowering target (see [backends.md](backends.md)) |
| `--quantize` | off | PT2E-quantize before lowering (supported backends) |

### `broker` — pure router (run one, forever)

```bash
python -m mobile broker
```

| flag | default | meaning |
|------|---------|---------|
| `--host` | `0.0.0.0` | informational; the broker binds all interfaces |
| `--job-port` | `15554` | frontend: feeders connect here |
| `--client-port` | `15555` | backend: workers connect here |
| `--ctrl-port` | `15556` | STOP broadcast |
| `--heartbeat` | `3.0` | seconds between status lines (with `-v`) |
| `-v, --verbose` | off | print routing/heartbeat status |

### `feed` — stream the corpus + diff + tally

```bash
python -m mobile feed --corpus corpus/portable                 # bulk → TSV + tally
python -m mobile feed w33:531 --corpus corpus/portable          # one job_id → JSON
python -m mobile feed path/to/w33_531.job                       # one .job → JSON
```

| flag | default | meaning |
|------|---------|---------|
| `jobs...` | — | specific job_id(s) or `.job` path(s) to replay (→ JSON); omit to feed the whole corpus (→ TSV) |
| `--corpus` | `tmp/corpus` | corpus dir (also resolves bare job_ids) |
| `--host` | `127.0.0.1` | broker host |
| `--job-port` | `15554` | broker frontend port |
| `--ctrl-port` | `15556` | broker ctrl port |
| `--from-tsv` | — | also take job_ids from a results TSV (col 2) |
| `--status` | — | with `--from-tsv`: keep only rows of this status |
| `-n, --graphs` | `0` | max jobs to feed; 0 = whole corpus |
| `--timeout` | `30.0` | per-job timeout → TIMEOUT |
| `--window` | `64` | max in-flight jobs (backpressure) |
| `--skip-log` | `tmp/skip_reasons_mobile.tsv` | failing rows: status + job_id + reason + op-chain |
| `--heartbeat` | `3.0` | seconds between feed status lines |
| `-v, --verbose` | off | verbose output |

### `client` — executor worker (one or more / per device)

```bash
python -m mobile client --host <broker-ip>
```

| flag | default | meaning |
|------|---------|---------|
| `--host` | `127.0.0.1` | broker host |
| `--client-port` | `15555` | broker backend port |
| `--ctrl-port` | `15556` | broker ctrl port |
| `--label` | `local` | label for this worker (e.g. `device`) |
| `--job-timeout` | `30.0` | watchdog: kill + TIMEOUT a job exceeding this |

## Fleet scripts

### `pregen_fleet.py` — parallel generation supervisor

```bash
python mobile/pregen_fleet.py --workers 96 --total 100000 --out corpus/portable --backend portable
```

| flag | default | meaning |
|------|---------|---------|
| `--workers` | `128` | parallel pregen workers |
| `--per-worker` | `100` | jobs each worker generates |
| `--total` | `0` | if set, split this many jobs across workers (overrides `--per-worker`) |
| `--out` | `tmp/corpus` | corpus dir |
| `--nodes` | `16` | real-op nodes per graph |
| `--leaf-prob` | `0.3` | leaf-vs-connect probability |
| `--out-alias-prob` | `0.1` | `out=` aliasing probability |
| `--seed` | `0xC0FFEE` | global seed |
| `--backend` | `portable` | lowering target for every worker |
| `--quantize` | off | PT2E quantize |
| `--stagger` | `0.3` | seconds between launches |
| `--worker-timeout` | `300.0` | kill+skip+respawn a worker idle (no stdout) this long — the **hang watchdog** |
| `--logdir` | `tmp/pregen_logs` | per-worker logs (`wI.log`) |

### `local_fleet.py` — executor supervisor

```bash
python mobile/local_fleet.py --clients 16 --corpus corpus/portable --host 127.0.0.1
```

Spawns + auto-respawns a feeder and a fleet of clients. Key flags: `--clients`,
`--corpus` (omit to manage clients only and run `feed` yourself), `--host`. See
`--help` for the rest.

### `run_pregen.sh` — turnkey pregen launcher

```bash
mobile/run_pregen.sh [BACKEND] [WORKERS] [TOTAL] [OUT]
mobile/run_pregen.sh qualcomm 24 100000
NODES=4 mobile/run_pregen.sh xnnpack 32 50000
```

Positional: `BACKEND WORKERS TOTAL OUT`. Env: `NODES` overrides the node count. Guards
against launching a second fleet while one is running.

## A full local run

```bash
# 1) generate
python mobile/pregen_fleet.py --workers 96 --total 100000 --out corpus/portable --backend portable

# 2) broker (separate terminal, runs forever)
python -m mobile broker -v

# 3) feed + clients (separate terminal, supervised)
python mobile/local_fleet.py --clients 16 --corpus corpus/portable --host 127.0.0.1
```

## Output files

| path | contents |
|------|----------|
| `corpus/<out>/<id>_<idx>.py` | every generated graph as a standalone runnable script (the repro) |
| `corpus/<out>/<id>/<id>_<idx>.job` | the wire payload (pte + inputs + eager + desc) |
| `corpus/<out>/_crashes/` | in-flight markers, `.skip` lists, preserved crash/hang samples |
| `tmp/skip_reasons_mobile.tsv` | every non-OK result: `status ⇥ job_id ⇥ reason ⇥ graph` |
| `tmp/mobile_samples/` | failing graph per MISMATCH/CRASH/TIMEOUT as `<status>_<job_id>.py` |
| `tmp/pregen_logs/wI.log` | per-worker generation logs |

## Reproducing one graph

A verdict references a `job_id` like `p0:12` → `corpus/<out>/p0_12.py` is the exact,
standalone, runnable graph. Or replay it through the live path:

```bash
python -m mobile feed p0:12 --corpus corpus/<out>     # → rich JSON observation
```
