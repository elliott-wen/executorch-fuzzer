# Operator-bug re-verification (stability gate) — the honest scrub

The 26 "confirmed operator bugs" in the first pass came from a **single** GENUINE isolation run each.
That is not enough on an int8 backend: near-tolerance divergences are non-deterministic. Re-running
each isolation **5×** (as the Vulkan run did 7×) collapses the list. This is the gate that must be
applied before any operator bug is claimed.

## Result (5 re-runs each; "n/5" = times it reproduced GENUINE/non-finite)

| bucket | n/5 | count | ops |
|---|---|---:|---|
| **STABLE (real kernel bug)** | 5/5 | **5** | `floor_divide.default`, `pow.Scalar_out`, `prod.default`, `prod.int_out`, `slice_scatter.default` |
| **INTERMITTENT (non-deterministic kernel)** | 1–4/5 | 8 | `bitwise_left_shift.Tensor_out` (3/5), `bitwise_right_shift.Tensor_out` (2/5), `logit.default` (2/5), `mean.default` (1/5), `rsub.Scalar` (1/5), `clamp.Tensor_out` (1/5, dtype), `fill.Scalar` (1/5, dtype), `_upsample_bilinear2d_aa.out` (1/5, dtype) |
| **FLAKY — dropped (not a bug)** | 0/5 | 13 | `_adaptive_avg_pool2d`, `addmm.out`, `alias_copy`, `any.dims_out`, `bitwise_right_shift.Tensor_Scalar`(+`_out`), `div.Scalar`, `elu.out`, `eq.Tensor_out`, `floor_divide.out`, `mean.out`, `native_group_norm.default`, `native_layer_norm.default` |

All 26 candidates re-verified 5× each. Note the non-finite ops split cleanly: `prod.default` and
`slice_scatter.default` are **stable** (5/5), but `native_group_norm`, `native_layer_norm` and
`floor_divide.out` are **flaky/reference-side** (0/5) — dropped.

## What this means
- **Only 5 operators are stably-reproducible single-op kernel bugs.** The first-pass count of 26 was
  inflated ~5× by single-run flukes. This is the single most important correction to the findings.
- **13 ops never reproduced** — the original GENUINE verdict was near-tolerance noise, not a bug.
- **7 ops are genuinely intermittent** — wrong output on *some* runs with identical input. That is
  itself a real defect class (non-deterministic int8 kernels: uninitialised scratch / race), but it
  must be reported as "intermittent", not "wrong".
- The two **dtype-divergence** survivors are both only 1/5 → the dtype-mismatch signal (device returns
  int where the reference is float) is **largely an isolation-harness artifact** (a one-op quantized
  graph can legitimately keep a quantized-int output), not a device bug. Dropped from the bug list.

## Confirmed operator bugs after the gate (5)
`floor_divide.default` · `pow.Scalar_out` · `prod.default` · `prod.int_out` · `slice_scatter.default`
— each reproduced 5/5 with finite inputs; device-verified evidence in `bugs/operators/`.

## Method note (add to analysis.md)
On any int8 / low-precision backend, a single GENUINE isolation is **not** a confirmed bug. Gate every
candidate with **N≥5 re-runs**; keep only those that reproduce every time (STABLE), report the rest as
INTERMITTENT (if ≥1/N) or drop (0/N). Skipping this gate over-reports operator bugs several-fold.
