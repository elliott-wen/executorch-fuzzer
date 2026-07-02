# Portable corpus — CRASH reasons (native aborts)

Systematic census of all **5,996 CRASH verdicts** from the portable differential-fuzz run.
A CRASH is a *native abort* that takes the executor process down (the feeder only records
"executor died (native abort)" — no op). To recover the actual reason, a **crash localizer**
(`tmp/run_portable/crash_localize.py`) re-ran each job's stored `.pte` in an isolated
subprocess and captured the fatal signal + the abort message (the firing `[file.cpp:line]`
ET_CHECK/assert), then normalized and aggregated them. Reproduction rate: **5,995 / 5,996**
aborted again (1 timeout) — crashes are deterministic.

## Distribution

| signal | count | share | reason | file |
|---|---:|---:|---|---|
| SIGABRT | 5,787 | 96.5% | fatal `ET_CHECK`/assert (mostly `tensor_impl.h size()` dim-out-of-range) | — |
| **SIGFPE** | **205** | **3.4%** | **integer divide/modulo by zero (hardware trap, no message)** | [03-integer-divide-by-zero-sigfpe.md](03-integer-divide-by-zero-sigfpe.md) |
| SIGSEGV | 3 | 0.1% | memory-safety / unhandled complex dtype | [07](07-complexfloat-elementwise.md), [08](08-memory-corruption.md) |
| TIMEOUT | 1 | 0.0% | did not finish in 25s | — |

## Deduplicated crash REASONS

| count | reason | file |
|---:|---|---|
| **4,043** | `narrow_copy` (and `select`) with a **negative / out-of-range `dim`** → `tensor_impl.h size()` asserts (`range [0,N] got -1`) | [01-narrow_copy-negative-dim.md](01-narrow_copy-negative-dim.md) |
| **1,740** | `unfold_copy` (and dim-ops) on a **0-D scalar input** → `tensor_impl.h size()` asserts (`range [0,-1] got 0`) | [02-unfold_copy-zero-dim.md](02-unfold_copy-zero-dim.md) |
| **205** | **integer `div`/`remainder`/`fmod`/`floor_divide` by zero** → SIGFPE | [03-integer-divide-by-zero-sigfpe.md](03-integer-divide-by-zero-sigfpe.md) |
| 13 | `tensor_util.h` **scalar_type mismatch** fatal Check (e.g. `{Half, Long}`) | [05](05-scalar-type-mismatch.md) |
| 10 | `reduce_util.cpp` path → **SIGFPE** (verified; not SIGABRT — int divide-by-zero) | [06](06-reduce-util-unhandled-dtype.md) |
| ~5 | **memory corruption** — `double free` / `free(): invalid next size` / SIGSEGV | [08](08-memory-corruption.md) |
| 2 | unhandled `ComplexFloat` (from `_fft_r2c`) into generic elementwise → SIGSEGV | [07](07-complexfloat-elementwise.md) |

> The two `tensor_impl.h size()` asserts (5,783 = **96.5%** of all crashes) attribute by op
> presence to **narrow_copy (4,257)** and **unfold_copy (1,527)** — the two kernels whose
> argument validation calls `in.size(dim)` *before* normalizing/guarding `dim`, so a fatal
> `ET_CHECK` in `TensorImpl::size()` fires instead of a graceful `InvalidArgument`.

## Key takeaways

1. **96.5% of all crashes are two missing-guard bugs** (`narrow_copy` negative-dim,
   `unfold_copy` 0-D). Both have full kernel root causes in
   [../crash-narrow_copy.md](../crash-narrow_copy.md) and
   [../crash-unfold_copy.md](../crash-unfold_copy.md). Fix: normalize+range-check `dim`
   (and add a 0-D guard) *before* any `in.size(dim)` call.
2. **A distinct third class — integer divide-by-zero → SIGFPE (205).** ExecuTorch portable
   does a raw integer `/`/`%`, which hardware-traps on a zero divisor where eager raises a
   catchable error. New finding.
3. **A long tail of ~30 includes 5 memory-safety crashes** (heap corruption / SIGSEGV) — the
   most severe, since they are out-of-bounds writes rather than asserts.

## Reproduce any crash
Every crash file now ships a self-contained `replay_*.py` (runs the stored `.pte` in a child
subprocess and exits 0 iff the crash reproduces). Run e.g.
`.venv/bin/python findings/portable/crash/replay_01-narrow_copy-negative-dim.py`. Or replay
any job directly:
```bash
.venv/bin/python tmp/run_portable/repro_one.py corpus/portable/<w>/<w>_<n>.job; echo "exit=$?"
# 134 = SIGABRT (assert) · 136 = SIGFPE (int div/0) · 139 = SIGSEGV (memory)
```
Data: `tmp/run_portable/crashes.jsonl` (per-job signal+signature), aggregate via
`tmp/run_portable/aggregate_crashes.py`.
