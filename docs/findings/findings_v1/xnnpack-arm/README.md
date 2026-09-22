# XNNPACK ARM (on-device) differential-fuzz — findings

First **on-device** run: `corpus/xnnpack` (100,032 graphs) executed on a real Snapdragon phone
via the rebuilt `executorch.aar`, jobs dispatched from the host broker over a `rathole` tunnel,
each diffed against eager PyTorch. Compared job-by-job against the **x86 host** XNNPACK run
(same corpus) to isolate what's ARM/NEON-specific.

## Verdicts (ARM vs x86)

| verdict | ARM | x86 |
|---|---:|---:|
| OK | 48,110 | 49,521 |
| MISMATCH | **14,859** | 11,024 |
| CRASH | 5,048 | 5,157 |
| SKIP | 32,015 | 34,330 |

ARM mismatches **+35%** over x86. Full comparison + confusion matrix: [00-distribution.md](00-distribution.md).

## One-line story

The on-device toolchain works end-to-end (the rebuilt AAR with the int16/bool/bfloat16 dtype
round-trip, the `jni_layer` exception→SKIP, fork-isolated CRASH). **Crashes and skips are
essentially all shared with x86/portable**; mismatches split into **8,743 shared** (the
documented x86/portable bugs) and **6,116 ARM-only**, of which the localizer confirms **81% are
genuinely NEON-specific** (x86 produces no divergence on those jobs).

## The ARM-specific finding → [bugs/neon-vlog-special-values.md](bugs/neon-vlog-special-values.md)

**XNNPACK's NEON `f32-vlog` microkernel drops Inf/NaN/x≤0 special-value handling** — it runs a
bounded rational polynomial, so `log(+Inf)` returns the finite constant **`88.376266`** instead
of `Inf`/`NaN` (the x86 scalar kernel calls libm `logf` and is correct). This drives the
dominant ARM-only cluster: `logit` (= `log(x/(1-x))`, ~2,385), `log` (~458), and the
`floor_divide`/`layer_norm` carriers (~3,036 ARM-only non-finite total). Confirmed by phone
replay (`w0:1114`: eager `[Inf,Inf]` vs ARM `[88.376266,…]`) and the x86 localizer (no
divergence). Real correctness divergence, not fast-math rounding.

## What's shared (no new work — reuse existing findings)

- **Crashes — 4,937 shared** (`narrow_copy` neg-dim, `unfold_copy` 0-D, integer `/0` SIGFPE) → [../portable/crash/](../portable/crash/README.md). **111 are unique to ARM** — but on-device replay showed they're the *same* shared `narrow_copy`/`unfold_copy` bug, just *reached* on ARM: x86 returns a graceful error `0x1` early (from `_log_softmax`/`add.Scalar`) that halts the `.pte` before the crash, while ARM proceeds and aborts. Not a new crash class. See [00-distribution.md](00-distribution.md).
- **Shared mismatches** (`select_scatter` dtype, shifts, `remainder`, `sum`, `sign`, `batch_norm`, and the activation NaN-laundering relu/clamp/min/max/exp/sqrt — *shared* because NEON and SSE both use min/max that drop NaN) → [../xnnpack/bugs/](../xnnpack/bugs/README.md), [../portable/bugs/](../portable/bugs/README.md)
- **Skips** — same kernel guards; the phone just reports coarser error codes (0x12/0x1/0x14). One phone-client-side gap: 53 `unsupported input dtype code`.

## Method note (why the localizer still helped for an on-device run)

The first-divergence localizer lowers graphs through **x86** XNNPACK on the host, so it can't
reproduce the phone's NEON kernels directly. But run over the **ARM-only** mismatch set it acts
as a discriminator: "no divergence on x86" ⇒ the divergence is ARM-specific. That gave the
**81% NEON-specific** confirmation. For actual ARM values we **replayed individual jobs back to
the phone** (`python feed.py <job_id> --corpus corpus/xnnpack`), which returns the device's real
output — that's how the `88.376266` magic constant was captured. Shared-mismatch root ops came
from the already-computed x86 localizer output.

Raw data + scripts: `tmp/run_xnnpack_arm/` (`skip_reasons.tsv`, `arm_only.tsv`,
`localized_arm_only.jsonl`, `compare_arm_x86.py`); `tmp/run_xnnpack/` for the x86 baseline.
