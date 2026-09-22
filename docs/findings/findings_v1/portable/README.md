# Portable-corpus differential-fuzz — findings

Run of the **portable** `.pte` corpus (`corpus/portable/`, 117,288 graphs) through
`xnnpack_client` — the in-process ExecuTorch host runtime that executes both portable-
and xnnpack-lowered programs. Each graph's output tensors are diffed against eager
PyTorch by the feeder. Raw data: `tmp/run_portable/` (`skip_reasons_portable.tsv`,
`feeder_final.log`, `analyze.py`, `gen_report.py`, `repro_one.py`).

## Verdict distribution → [00-distribution.md](00-distribution.md)

| verdict | count | share |
|---|---:|---:|
| OK | 63,480 | 54.1% |
| MISMATCH | 10,066 | 8.6% |
| CRASH | 5,996 | 5.1% |
| SKIP | 37,746 | 32.2% |
| TIMEOUT | 0 | 0.0% |

Of programs that *ran to completion* (73,546), **13.7% mismatched eager**.

## Per-verdict reason directories (systematic, deduplicated)

- **[crash/](crash/README.md)** — all 5,996 crashes localized to their fatal abort: 96.5% are
  `narrow_copy` negative-dim + `unfold_copy` 0-D, plus a new **integer divide-by-zero SIGFPE**
  class (205) and ~5 memory-safety crashes.
- **[skip/](skip/README.md)** — all 37,746 graceful refusals categorized: dtype-coverage gaps,
  unimplemented arg paths, input-validation guards, type-promotion guards, hard-disabled paths.
- **[bugs/](bugs/README.md)** — source-level per-bug deep-dives for the MISMATCH defects.

Every distinct bug (MISMATCH + CRASH) is now split into its own write-up with a **self-contained
`replay_<slug>.py`** that triggers it and exits 0 iff it reproduces. → **[REPLAY.md](REPLAY.md)**
indexes all 22 replay scripts and how to run them.

(The methodology — a node-level **first-divergence localizer** that attributes every mismatch
to a born-here root op, and a **crash localizer** that captures each fatal abort signature —
is in `tmp/run_portable/` as `localize.py` / `crash_localize.py` + their aggregators.)

## Reports

| report | cluster | size | headline | verdict |
|---|---|---:|---|---|
| [crash-narrow_copy.md](crash-narrow_copy.md) | CRASH `narrow_copy` | 3,758 | negative / 0-D `dim` → `in.size(dim)` is checked **before** normalization → `tensor_impl.h:136` fatal assert → SIGABRT | **REAL ET BUG** |
| [crash-unfold_copy.md](crash-unfold_copy.md) | CRASH `unfold_copy` | 2,116 | 0-D scalar input → `self.size(0)` on a 0-D tensor → same fatal assert → SIGABRT | **REAL ET BUG** |
| [mismatch-dtype.md](mismatch-dtype.md) | MISMATCH dtype | 1,212 | `select_scatter` decomposes to `where` in `to_edge()` **without casting src→self.dtype** → wrong output dtype (single root cause for 100%) | **REAL ET/EXPORT BUG** |
| [mismatch-delta-exact.md](mismatch-delta-exact.md) | MISMATCH exact-Δ (rtol=0) | ~3,202 | integer-semantics: real `remainder` sign + `sum` cast-order bugs (~27%); negative-shift UB + int64 overflow (~28%); rest inherited/structural | **MIXED — real bugs + UB** |
| [mismatch-delta-float.md](mismatch-delta-float.md) | MISMATCH float-Δ (rtol>0) | ~2,177 | ~half real: integer-into-float `bitwise_left_shift` + div-by-zero (huge Δ are 2^56..2^63, not fp16 overflow); ~half benign fp16 precision | **MIXED — ~50% real** |
| [mismatch-non-finite.md](mismatch-non-finite.md) | MISMATCH nan/inf | 2,937 | real: portable `sign(NaN)≠0`, negative bit-shift UB → fp16 overflow; artifacts: upstream OOB NaN re-positioned, fp16 singularities | **MIXED** |
| [mismatch-shape.md](mismatch-shape.md) | MISMATCH shape | 485 | eager resizes aliased `out=` buffers in place (topk/min.dim/max.dim), portable's static planning never resizes and **silently returns the wrong shape** | **BEHAVIORAL DIVERGENCE — not a kernel bug, but a real execution-model difference + missing runtime check** |
| [skip-analysis.md](skip-analysis.md) | SKIP | 37,746 | graceful kernel refusals — dtype/arg/shape coverage gaps (not bugs); `_conj_physical` unhandled-dtype alone = 10k | **COVERAGE MAP** |

## Source-level per-bug deep-dives → [bugs/](bugs/README.md)

Each MISMATCH bug traced to the exact offending line (portable kernel C++ / export decomp /
`torch._refs`), with a minimal repro and fix:
[select_scatter dtype](bugs/select_scatter-dtype.md) ·
[remainder sign](bugs/remainder-sign.md) ·
[sum dtype-loss](bugs/sum-cast-order.md) ·
[sign(NaN)](bugs/sign-nan.md) ·
[floor_divide by-zero](bugs/floor_divide.md) ·
[bitwise shift](bugs/bitwise-shift-negative.md) ·
[distance p=0 NaN](bugs/distance-nonfinite.md) ·
[batch_norm var=0/eps=0](bugs/normalization-nonfinite.md) ·
[maxpool2d backward OOB](bugs/maxpool2d-backward.md). Newly split out of the cluster reports
(each with a replay): [prod/pow overflow](bugs/prod-pow-int64-overflow.md) ·
[reduction into bool out](bugs/reduction-into-bool-out.md) ·
[shape aliased-out resize](bugs/shape-aliased-out-resize.md) ·
[comparison/bool flips](bugs/comparison-bool-flips.md) ·
[transcendental fp16 precision](bugs/transcendental-fp16-precision.md) ·
[non-finite inherited](bugs/nonfinite-inherited-reposition.md). The [bugs/README.md](bugs/README.md)
index ranks them by severity and lists the non-bug artifacts. Two earlier cluster-level diagnoses
were corrected there (`sum` is an export functionalization dtype-loss, not the kernel;
`floor_divide` integer floor is correct — only float `0/0` diverges).

## Top actionable findings (real ExecuTorch defects)

1. **`narrow_copy` aborts on negative/out-of-range `dim`** (3,758 crashes). `check_narrow_copy_args`
   calls `in.size(dim)` with the raw (un-normalized) `dim` *before* `op_narrow_copy.cpp`
   normalizes it; `TensorImpl::size()` fatally asserts on `dim<0`. **Fix:** normalize +
   range-check `dim` before any `in.size(dim)` call.
2. **`unfold_copy` aborts on 0-D input** (2,116 crashes). `check_unfold_copy_args` accepts
   `dim==0` for a 0-D tensor then calls `self.size(0)`, which fatally asserts. **Fix:** add a
   0-D guard (eager treats a 0-D input as 1-D and returns a `[0]`/`[1]`-shaped result).
3. **`select_scatter` dtype-promotion bug** (1,212 dtype mismatches, all of them). The
   `to_edge()` decomposition lowers `select_scatter(self, src)` to `where(mask, src, self)`
   without casting `src` to `self.dtype`, so the program output dtype becomes
   `result_type(self, src)` instead of eager's `self.dtype`. **Fix:** cast `src→self.dtype`
   in the decomposition; audit sibling `*_scatter` decomps.
4. **Integer-semantics kernel bugs** (subset of the delta clusters): portable `remainder`
   uses `fmod`-style (dividend) sign instead of PyTorch's divisor sign; `sum`/reduction
   into an int64 `out` accumulates-then-casts instead of casts-then-accumulates; portable
   `sign(NaN)` returns NaN instead of 0. These produce wrong *values*, not just precision noise.
5. **Negative bit-shift** (`bitwise_left_shift`/`right_shift` with negative count) is C++ UB;
   portable and eager diverge (eager clamps to 0). Largely generator-fed, but a real
   divergence — either define the behavior or have the kernel guard it.

## Known non-bugs (harness / generator artifacts — fix the fuzzer, not the backend)

- **All 485 shape mismatches** — the differential generator reuses a node's pre-allocated
  `out=` buffer as a later multi-output reduction's output; eager resizes it in place,
  portable's static planning cannot. Don't alias out-buffers across nodes.
- **Most large fp16 deltas mis-bucketed** — the harness buckets by absolute `|delta|`, so a
  real `eager≈0` bug lands in the "small" bucket while benign fp16 noise can look large.
  Recommend bucketing by **relative** error and scaling tolerance by output **dtype**.

## Reproduce any single job

```bash
cd /data/jwen929/mobile
.venv/bin/python tmp/run_portable/repro_one.py corpus/portable/<w>/<w>_<n>.job 2>&1; echo "exit=$?"
# exit 134=SIGABRT / 139=SIGSEGV → the CRASH; "RAN OK; outputs:" → ran (for MISMATCH value diffing)
```
Value-diff harnesses built during analysis: `tmp/run_portable/cmp_delta.py`,
`tmp/run_portable/cmp_deltaf.py`, `tmp/run_portable/cmp_nonfinite.py`.

## How to re-run the whole sweep

```bash
cd /data/jwen929/mobile
PYTHONPATH=/data/jwen929 .venv/bin/python -m mobile broker -v &           # 1. broker
.venv/bin/python local_fleet.py --clients 64 --backend xnnpack \           # 2. fleet + one-shot feeder
    --corpus corpus/portable --host 127.0.0.1 --logdir tmp/run_portable/logs
# failing rows land in tmp/skip_reasons_mobile.tsv; regenerate reports with gen_report.py
```
