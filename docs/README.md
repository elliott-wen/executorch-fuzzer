# mobile fuzzer — documentation

`mobile/` is a self-contained **eager-vs-ExecuTorch differential fuzzer**. It
generates multi-op PyTorch graphs, lowers each to an ExecuTorch program (`.pte`),
runs that `.pte` on the ExecuTorch runtime, and compares the result against eager
PyTorch (the oracle). Any divergence is a bug in the lowering/runtime stack.

These docs describe the whole pipeline. Read them in order for a tour, or jump to
the subsystem you need.

| # | Doc | What it covers |
|---|-----|----------------|
| 0 | [environment-setup.md](environment-setup.md) | Python 3.12 venv, `requirements.txt`, the CPU-only / no-CUDA discipline, the device/Android toolchain |
| 1 | [overview.md](overview.md) | What the tool is, the differential-testing idea, verdicts, core design principles |
| 2 | [architecture.md](architecture.md) | End-to-end data flow, the ZeroMQ broker topology, crash isolation |
| 3 | [graph-generation.md](graph-generation.md) | How a graph is built: the Z3 op model, the seed schedule, growth & anchoring, the adapter ladder, `out=` aliasing |
| 4 | [backends.md](backends.md) | The backend registry, each lowering target, quantization, compile-vs-execute (incl. Apple CoreML/MPS) |
| 5 | [corpus-and-pregen.md](corpus-and-pregen.md) | Pre-generation, the parallel fleet, resume/skip, the crash + hang watchdog, determinism |
| 6 | [networking.md](networking.md) | The broker / feed / client protocol, scaling, phone clients |
| 7 | [cli-reference.md](cli-reference.md) | Every `python -m mobile` subcommand and flag, the fleet scripts, file layout, env |

## TL;DR

```bash
cd /data/jwen929/pytorch
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=""

# 1) Generate a corpus to disk (offline, parallel).
python mobile/pregen_fleet.py --workers 96 --total 100000 --out corpus/portable --backend portable

# 2) Start the broker (one, runs forever).
python -m mobile broker

# 3) Feed the corpus + run a client fleet (supervised).
python mobile/local_fleet.py --clients 16 --corpus corpus/portable --host 127.0.0.1
```

## Package map

```
mobile/
  __main__.py            CLI dispatch (pregen | broker | feed | client)
  pregen_fleet.py        parallel pre-generation supervisor (+ hang watchdog)
  local_fleet.py         feeder + executor-client supervisor
  run_pregen.sh          turnkey pregen launcher
  gen/                   GRAPH GENERATION
    opnode.py              per-op Z3 constraint unit
    graph_build.py         DAG construction (seed + grow + anchoring)
    adapter.py             port-connection glue (cast/broadcast/reshape/slice/pad/clamp)
    graph_ir.py            the graph data structure + source emitter
    emit.py                argument-source rendering
    concretize.py          Z3 model → concrete tensors
    export_et.py           eager oracle + torch.export → to_executorch
    backends/              per-backend lowering (portable/xnnpack/vulkan/arm/qnn/coreml/mps)
    runnable_ops.py        the usable op set
    executorch_allowlist.py / blocklist.py   op inclusion rules
  net/                   TRANSPORT
    protocol.py            language-neutral binary wire codec
    corpus.py              on-disk job store
    broker.py              pure ZeroMQ router
    feed.py                stream corpus + own the diff/tally
    client.py              executor worker (runs .pte)
    et_runner.py           single-.pte runner
    compare.py             eager-vs-ET comparison
    pregen.py              single-worker corpus generator
  gen/z3engine/          VENDORED Z3 CONSTRAINT SOLVER (was top-level engine/)
    model.py               symbolic type model
    sampler.py             diverse valid-input sampler (was test_valid_inputs.py; library, no CLI)
    op_lookup/solver/schema_overrides/entry_points_data
```

> The canonical, code-level source of truth is each module's docstring. These docs
> summarize and connect them; when in doubt, read the module.
