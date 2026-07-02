# XNNPACK ARM (on-device) vs x86 (host) — distribution + comparison

Same corpus (`corpus/xnnpack`, 100,032 graphs), run two ways and each diffed against eager
PyTorch:
- **ARM phone** — the rebuilt `executorch.aar` on a real Snapdragon, jobs dispatched from the
  host broker over a `rathole` tunnel (`tmp/run_xnnpack_arm/skip_reasons.tsv`).
- **x86 host** — `xnnpack_client` in-process (`tmp/run_xnnpack/skip_reasons_xnnpack.tsv`).

Because it's the **same corpus** (same job_ids, same graphs, same eager reference), the two runs
compare job-by-job.

> ## ⚠ Methodological caveat: the two runs use DIFFERENT ExecuTorch builds
> The x86 baseline runs the **pip wheel `executorch 1.4.0.dev20260625`** (in `.venv`); the ARM
> phone runs our **source-built AAR from `pytorch_ref/executorch @ 2759ef1`**. They are *not the
> same build*, so an ARM-vs-x86 difference can be **architecture** (NEON vs SSE/scalar) **or
> ExecuTorch version** (different portable-kernel source). Confirmed example: `op_log_softmax.cpp`
> — the **wheel has a `Check(false)` hard-disable at line 152** (→ x86 SKIPs), the **AAR source
> fully implements it** (→ ARM runs it) — which is what gates the "ARM-unique crash" class below.
> **What's safely arch:** the XNNPACK **delegate microkernel** findings (`f32-vlog`, gelu/tan/erf)
> — those are XNNPACK's own arch-selected kernels and the source was read to confirm the NEON-vs-
> scalar split. **What's version-confounded:** portable-kernel verdict diffs (e.g. the
> `_log_softmax`/`add.Scalar` early-error gating). A clean pure-arch comparison would rebuild the
> x86 baseline from the *same* `@2759ef1` source.

## Top-level verdicts

| verdict | x86 | ARM | Δ |
|---|---:|---:|---:|
| OK | 49,521 | 48,110 | −1,411 |
| MISMATCH | 11,024 | **14,859** | **+3,835** |
| CRASH | 5,157 | 5,048 | −109 |
| SKIP | 34,330 | 32,015 | −2,315 |
| TIMEOUT | 0 | 0 | |

**ARM mismatches 35% more than x86** — the NEON microkernels add divergences x86's SSE/scalar
kernels don't have.

## Confusion matrix (rows = x86 verdict, cols = ARM verdict)

| x86 ＼ ARM | OK | SKIP | CRASH | MISMATCH |
|---|---:|---:|---:|---:|
| **OK** | 43,791 | 216 | 1 | **5,513** |
| **SKIP** | 1,995 | 31,649 | 109 | 577 |
| **CRASH** | 102 | 92 | 4,937 | 26 |
| **MISMATCH** | 2,222 | 58 | 1 | 8,743 |

The divergence sets are **not symmetric**:
- **5,513 jobs OK on x86 → MISMATCH on ARM** — ARM-specific divergences (the new finding).
- **2,222 jobs MISMATCH on x86 → OK on ARM** — ARM computes these *correctly* where x86 didn't
  (e.g. x86 fast-math cases ARM happens to get right).
- **8,743 mismatch on both** — shared bugs.
- CRASH (4,937) and SKIP (31,649) are almost entirely shared (non-delegated portable kernels +
  the same kernel guards).

## Mismatch split: shared vs ARM-only

ARM mismatches = **8,743 shared** + **6,116 ARM-only**.

### Shared (8,743) → the already-documented x86/portable bugs
Attributed via the existing x86 first-divergence localizer (`tmp/run_xnnpack/localized.jsonl`):
`bitwise_left_shift` 1,513 · `select_scatter` (dtype) 1,157 · `remainder` 891 · `sum` 558 ·
`sign` 408 · `prod` 325 · `broadcast_to` (shape) 273 · `_native_batch_norm` 167 · and the
**activation NaN-laundering** `exp` 158 / `sqrt` 127 / `minimum` 98 / `relu` 75 / `hardtanh` 69.
→ See [../xnnpack/bugs/](../xnnpack/bugs/README.md) and [../portable/bugs/](../portable/bugs/README.md);
the activation NaN-laundering is shared because **both** NEON and SSE clamp/minmax return the
non-NaN operand.

### ARM-only (6,116) → NEON-specific
Ran the localizer (x86 lowering) over all 6,116: **81% "no_reproduce"** (x86 produces no
node-level divergence) — confirming they are genuinely ARM-NEON-specific; 8% are borderline
sub-tolerance shared cases, 10% x86 export errors. Op breakdown of the ARM-only set:

| count | op | kind | note |
|---:|---|---|---|
| **1,544** | `logit` | non-finite | **dominant — NEON `f32-vlog` bug** (see below) |
| 223 | `bitwise_right_shift` | delta | NEON vs SSE shift-by-neg/oversize |
| 169 | `logical_xor` | delta | inherited |
| 159 | `sum` | delta | NEON accumulation order |
| 130 | `div` | non-finite | NEON division domain |
| 112 | `floor_divide` | non-finite | NEON by-zero, carrier |
| 110 | `log` | non-finite | NEON `f32-vlog` directly |
| 59 | `native_layer_norm` | non-finite | NEON transcendental carrier |
| … | `pow`/`mean`/`remainder`/`mul` | non-finite | NEON transcendental/division |

**More ARM-only ops (beyond the top table), not in any shared/documented finding:**
- **NEON transcendentals `gelu` (37), `tan` (37), `erf` (32)** — each has its own NEON
  microkernel. **Phone-replay confirmed (all ARM-only):** `gelu` (w0:1273) is **non-finite** —
  the same no-special-value class as [f32-vlog](bugs/neon-vlog-special-values.md);
  `tan` (w12:1473, Δ0.30) and `erf` (w10:1362, Δ0.84) are **value-deltas** — NEON
  lower-precision polynomials exceeding `atol=0.001` where x86 matches eager (fast-math, not the
  catastrophic Inf→finite log bug). So the NEON transcendental issue is two-pronged:
  no-special-value (log/logit) **and** reduced-precision approximation (tan/erf).
  `gelu` is a third flavor: replay (w0:1273) shows ARM **launders NaN→0** (out[4] device `0.0`
  vs eager `nan`) — like the shared relu/clamp family but here ARM-specific to gelu's NEON path.
- **NEON integer/boolean `bitwise_xor`/`or` (50/43), `logical_or` (49), `cumsum` (47)** — NEON
  integer/accumulation differences (delta).
- `where`/`copy`/`transpose_copy`/`view_copy`/`detach_copy` — mostly *inherited* structural carriers.

## The headline ARM-only bug → [bugs/neon-vlog-special-values.md](bugs/neon-vlog-special-values.md)

XNNPACK's **NEON `f32-vlog` microkernel has no Inf/NaN/x≤0 special-value handling** — it
bit-extracts the exponent and evaluates a bounded rational polynomial, so **`log(+Inf)` returns
the finite constant `88.376266`** instead of `+Inf`/`NaN`. The x86 scalar variant calls libm
`logf` and is correct. `logit(x) = log(x/(1-x))` decomposes onto this NEON `log` (logit isn't a
native XNNPACK op), so the whole logit cluster inherits it; `log`/`floor_divide`/`layer_norm`
non-finites are direct or downstream carriers. **Phone-replay evidence:** `w0:1114` →
eager `[Inf, Inf]` vs ARM `[88.376266, 88.376266]`, x86 == eager. Scope ≈ 3,036 ARM-only
non-finite mismatches. Source: `XNNPACK/src/f32-vlog/gen/f32-vlog-neon-rational-3-3-div.c` vs
`f32-vlog-scalar-log.c:34`; arch selection `src/configs/unary-elementwise-config.c:1429-1431`.

## CRASH (5,048) — mostly shared, but a UNIQUE ARM class

4,937 are the **same jobs that crash on x86** — `narrow_copy` negative-dim + `unfold_copy` 0-D
`tensor_impl.h size()` aborts + integer divide-by-zero SIGFPE (phone reports
"executor process died (native abort)"). → [../portable/crash/](../portable/crash/README.md).

**UNIQUE to ARM — 111 crashes that x86 does NOT crash on.** On-device replay
(`feed.py w0:480` / `w0:1095`) showed these are **the same shared `narrow_copy`/`unfold_copy`
crash bug, just *reached* on ARM**: the graph contains that crash-prone op AND an upstream op
(`_log_softmax`/`add.Scalar`) that **errors early on x86 but not ARM** — and that gating turned
out to be a **VERSION difference, not arch**: the x86 wheel's `op_log_softmax.cpp:152` has a
`Check failed (false)` hard-disable that fires at instruction 0:0 → halts the `.pte` → SKIP,
while the AAR-source `op_log_softmax.cpp` (`@2759ef1`) **fully implements** the op → it runs →
execution reaches the shared `narrow_copy`/`unfold_copy` crash. E.g. `w0:480` op-chain
`_log_softmax → narrow_copy×3 → floor_divide`; `w0:1095` has `unfold_copy`. So: **the crash is
the same shared bug; the ARM-vs-x86 difference is the `_log_softmax` portable-kernel version**
(see the caveat at top). My initial "graceful-check-turned-fatal on device" guess was wrong (the
replay + source disproved it). Sample: `w0:480 w0:1095 w1:570 w10:292`.
- Counterpart: **102 jobs CRASH on x86 but run fine (OK) on ARM** — ARM gets through them.

## SKIP (32,015) — same gaps, coarser messages
The phone (Android JNI) reports skips as Java exceptions with **error codes only**, not the
per-op native `[op_*.cpp] Check failed` text the host produces:

| count | reason | maps to |
|---:|---|---|
| 26,534 | `ExecutorchInvalidArgumentException: Error 0x12` | the kernel-refusal bulk (`_conj_physical`, `expand_copy`, `copy`, `group_norm`, … — same as x86, no per-op detail) |
| 3,516 | `ExecutorchRuntimeException: Error 0x1` | ≈ the XNNPACK **load-failure** class (x86 had 3,487 — see [../xnnpack/bugs/xnnpack-load-failure.md](../xnnpack/bugs/xnnpack-load-failure.md)) |
| 1,912 | `ExecutorchRuntimeException: Error 0x14` | delegate/runtime error class |
| 53 | `IllegalArgumentException: unsupported input dtype code` | **the Android client itself** can't build a rarer input dtype (a phone-client coverage gap, not a kernel issue) |

To get per-op detail for an ARM skip, cross-reference the job's op-chain (in the corpus / the
x86 TSV) — the underlying kernel guard is the same.

Raw data + scripts: `tmp/run_xnnpack_arm/` (`skip_reasons.tsv`, `arm_only.tsv`,
`localized_arm_only.jsonl`, `compare_arm_x86.py`).
