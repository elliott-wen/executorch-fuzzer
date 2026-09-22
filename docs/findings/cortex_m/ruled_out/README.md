# cortex-m — ruled-out suspects (filtered per analysis_single.md step 3)

Every non-OK graph is isolated to one operator by construction, but "isolated" ≠ "cortex-m bug".
These are the divergences that survive isolation yet are **not** cortex-m backend bugs, with the
filter that removed each. The filter-3a signal here is **which `cortex_m::` kernel the lowered
`.pte` actually calls** (cortex-m is pass-based, so the `delegated.ops` header is always 0 and
useless — see `_work/cm_kernels.py`).

## 3a — portable-compute MISMATCH (`NO-CM` and `CM-QUANT-ONLY`) — 1,428 of 1,436 MISMATCH
The op-under-test's **compute** ran as a portable `executorch` kernel; the only `cortex_m::` ops in
the graph were the `quantize`/`dequantize` boundary (`CM-QUANT-ONLY`) or none at all (`NO-CM`). A
value divergence is therefore a **portable-kernel or reference issue, not a cortex-m compute-kernel
bug** — the cortex-m analog of ethos-u's `ops=0` fallback. These may be genuine *portable* bugs, but
they are not filed against cortex-m.

Why the shared `cortex_m::quantize`/`dequantize` boundary is exonerated (not the cause of the
`CM-QUANT-ONLY` mismatches): **55,128 graphs returned OK**, thousands of them `CM-QUANT-ONLY`, all
going through the same quantize/dequantize kernels. A bug there would break them uniformly. Instead
the mismatches concentrate in specific portable **compute** ops, so the boundary is sound.

| MISMATCH count | operator | bucket | attribution |
|---:|----------|--------|-------------|
| 211 | `index_put` | CM-QUANT-ONLY | portable `index_put` kernel |
| 180 | `fill.Scalar` | CM-QUANT-ONLY | portable `fill` kernel |
| 103 | `max_pool2d_with_indices_backward` | CM-QUANT-ONLY | portable backward kernel |
| 30 | `where.self` | CM-QUANT-ONLY | portable `where` kernel |
| 27 | `div.out_mode` | CM-QUANT-ONLY | portable `div` (rounding-mode) kernel |
| 19 | `prod.int_out` | CM-QUANT-ONLY | portable `prod` kernel |
| 18 | `logit` | CM-QUANT-ONLY | portable `logit` kernel |
| 9 | `_upsample_bilinear2d_aa.out` | CM-QUANT-ONLY | portable upsample kernel |
| — | (others) | CM-QUANT-ONLY / NO-CM | portable |

## 3d — integer bitwise-shift UB (`NO-CM`) — negative / oversized shift
`bitwise_left_shift` / `bitwise_right_shift` (all overloads, ~600 MISMATCH) run as **portable
integer** kernels (`NO-CM`) on fuzzer-drawn shift operands that include **negative and ≥bitwidth**
counts — undefined behaviour in C/C++. The eager (PyTorch) and portable-kernel results diverge
because UB is implementation-defined, not because of a cortex-m defect. Low-confidence by
construction; ruled out. `remainder`/`fmod` mismatches are the same integer-portable class (division
/ sign-convention edge cases), not cortex-m.

## 3b — determinism
The confirmed `permute_copy → cortex_m::transpose` bug was gated at **N=5 re-runs of the same
`.pte`** through the feeder (stored quantized reference): **REPRO 5/5** for every instance. The
ruled-out portable mismatches were not individually determinism-gated because filter 3a already
removes them from cortex-m scope; several (int8 near-tolerance, shift UB) would additionally be
intermittent under 3b.

## Not a device bug — build/infra (see [../skips.md](../skips.md))
`linear` / `linear.out` / `_fft_r2c` CRASH (rc=5) and `_fft_r2c` TIMEOUT are **runner build
failures** (executorch portable-codegen empty-`#include`, no portable kernel), not device crashes.
The factory-op (`zeros/ones/full/arange`) and `stack` SKIPs are arm_executor_runner input-prep /
no-output limits, not cortex-m kernel rejections.
