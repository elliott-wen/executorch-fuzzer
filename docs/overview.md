# 1. Overview

## What it does

The mobile fuzzer answers one question at scale:

> Does lowering a PyTorch graph to ExecuTorch and running it on the ExecuTorch
> runtime produce the **same result** as running the graph in eager PyTorch?

It does this differentially:

1. **Generate** a random but *valid* multi-op graph (a small DAG of real aten ops).
2. **Run it eager** — this is the **oracle** (ground truth).
3. **Lower it** — `torch.export` → `to_executorch` → a `.pte` program.
4. **Run the `.pte`** on the ExecuTorch runtime (portable CPU kernels by default —
   the actual mobile execution stack, not inductor).
5. **Compare** the two outputs.

Eager and ExecuTorch should agree to within numerical tolerance. When they don't,
that divergence is a real bug in the export/lowering/runtime path — the signal the
fuzzer exists to find.

## Verdicts

Every job ends in exactly one verdict:

| verdict | meaning |
|---------|---------|
| **OK** | eager == ExecuTorch (within tolerance) |
| **MISMATCH** | eager ≠ ExecuTorch — a real divergence, the bug signal |
| **CRASH** | the worker died on the graph (native abort, or a hung kernel the watchdog killed) |
| **TIMEOUT** | the job never returned (dead/hung worker; broker backstop) |
| **SKIP** | the runtime couldn't run it (missing kernel / unhandled dtype) — logged with the real error |

A run is a tally over these, plus on-disk repros for every non-OK job.

## Why it finds real bugs (the validity discipline)

A differential fuzzer is only useful if a MISMATCH means a *bug*, not a *bad test*.
Three disciplines guarantee that:

- **Valid inputs only.** Each op's inputs are produced by a **Z3 constraint solver**
  that encodes the op's real preconditions (shape/dtype/value constraints). Graphs
  that would raise in eager are filtered out, not reported.
  See [graph-generation.md](graph-generation.md).
- **Determinism.** Inputs are materialized **once** at generation time and shipped
  byte-identical to the executor — there is no client-side RNG. Seeding is a pure
  function of the `job_id`, so a `job_id` always reproduces the exact same graph.
  RNG ops are excluded. → any MISMATCH is reproducible and real.
- **Overload-exact op set.** The op set is regenerated from the installed ExecuTorch
  runtime registry: an op is included **iff** its `.out` kernel is actually
  registered. So the fuzzer never generates an op the runtime can't run.
  See [backends.md](backends.md) and [graph-generation.md](graph-generation.md).

## Core design principles

- **Pre-generate, don't fuzz live.** Graphs are generated and lowered **once**, to a
  corpus on disk, then replayed. Generation (CPU-heavy, embarrassingly parallel) is
  decoupled from execution; runs are reproducible; every graph is on disk as a
  standalone runnable `.py` for debugging. See [corpus-and-pregen.md](corpus-and-pregen.md).
- **The broker is a pure router.** It moves job frames between feeders and workers and
  never touches a tensor. The **feeder** holds the eager reference and does the diff;
  the **worker** just runs the `.pte`. This keeps the broker trivial and lets the
  worker be anything — a local process today, a phone tomorrow. See [networking.md](networking.md).
- **Crash isolation.** A graph that natively aborts or hangs takes down only a
  disposable executor child; the coordinator reports CRASH/TIMEOUT and respawns.
  Nothing in the pipeline assumes a graph is safe to run.
- **CPU only.** `CUDA_VISIBLE_DEVICES=""` everywhere; portable CPU kernels are the
  faithful mobile stand-in.

## Self-contained

The `mobile/` package owns its entire pipeline — a vendored Z3 constraint solver
(`gen/z3engine/`), graph generation (`gen/`), and transport (`net/`). Its only external
dependencies are the `z3`, `torch`, and `executorch` libraries plus the
`approved_constraints/` data that encodes each op's preconditions.
