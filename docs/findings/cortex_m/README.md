# Cortex-M (Corstone-300 FVP) — single-operator differential-fuzz findings

Backend: **cortex-m** (int8 CMSIS-NN **operator library** on the Cortex-M55 core — no NPU, no
delegate), run on the Arm **Corstone-300 FVP** via `fvp_client`. Corpus: `corpus_v3/cortex-m` — a
**single-operator** corpus (`--nodes 1`: each graph is exactly one op wired to leaf inputs, so every
non-OK outcome is a minimal repro attributed to one operator). int8 (`quant=ALWAYS`); the compare is
**stored quantized reference vs device**, same quantization space. Method: [`analysis_single.md`](../../analysis_single.md).

## cortex-m is pass-based — the filter-3a adaptation
cortex-m is **not a delegate**: the CortexM pass PT2E-quantizes, then rewrites quantized aten ops
into `cortex_m::` CMSIS-NN custom ops (`ReplaceQuantNodesPass` + fusion/convert passes) and
serializes a portable `.pte`. So the corpus's `delegated.ops` counter is **structurally 0 on all
61,735 jobs** — filter 3a ("ops=0 ⇒ portable, out of scope") does not apply as written. It was
replaced with the real signal: **re-lower each non-OK job and read which `cortex_m::` kernels the
`.pte` calls** ([`_work/cm_kernels.py`](_work/cm_kernels.py)), bucketing every non-OK graph:
- **CM-COMPUTE** — the op-under-test became a `cortex_m::` compute kernel → in-scope cortex-m bug.
- **CM-QUANT-ONLY** — only the `cortex_m::quantize`/`dequantize` boundary is cortex_m; the compute op
  ran as a **portable** quantized kernel → a divergence is portable, not cortex-m (boundary exonerated
  because 55k OK graphs share it).
- **NO-CM** — no `cortex_m::` op at all → pure portable/reference issue.

**The cortex-m compute surface is barely exercised by a single-op corpus.** Only ~2 % of graphs get
a `cortex_m::` *compute* kernel — essentially `transpose` (from permute/pixel_shuffle/transpose_copy),
`minimum`, `quantized_mul`, `pad`. The multi-op quantizer patterns (conv+relu, linear+bias) that
would route conv/linear/add through CMSIS-NN never form in one-op graphs, so those run portable with
only the cortex_m quantize/dequantize boundary. This is the cortex-m analog of ethos-u's 89 % `ops=0`.

## Run summary
Full corpus fed through an FVP fleet + one dedicated broker.

| total | OK | MISMATCH | CRASH | TIMEOUT | SKIP |
|------:|---:|---------:|------:|--------:|-----:|
| 61,735 | 55,128 | 1,485 | **253** | **419** | 4,450 |

6,607 non-OK across 71 operators. Of the non-OK, 6,446 were bucket-classified by re-lowering (the
161 unclassified are the `linear`/`_fft_r2c` graphs whose `build_job` itself hangs — see below).

**All 253 CRASH and all 419 TIMEOUT are INFRA, not device faults:** they are `linear` / `linear.out`
/ `_fft_r2c` runner **build** failures (executorch portable-codegen emits an empty `#include` because
the op has no portable kernel — `aten::linear.out` has none). **Zero genuine device crashes were
observed across 61,735 graphs.** These three ops therefore have no device coverage (100 % build
attrition). See [skips.md](skips.md).

## Confirmed cortex-m operator bug
Device-verified, finite inputs, **deterministic (REPRO 5/5)** on the FVP, attributed to a `cortex_m::`
kernel by re-lowering. Repro + index in [bugs/](bugs/) · [bugs/INDEX.md](bugs/INDEX.md).

### MISMATCH — WRONG-VALUE
| operator | cortex_m kernel | mechanism | eager → device | prevalence |
|----------|-----------------|-----------|----------------|-----------|
| [`permute_copy`](bugs/permute-copy-transpose.md) | `cortex_m::transpose` | general/**identity** permute lowered to a fixed axis-swap → device returns the *transpose*, not the permutation (silent: same shape, wrong values) | `1.164 → -0.912` (max\|Δ\|=2.075) | 8/8 permute_copy MISM |

This is the **only** CM-COMPUTE MISMATCH in the entire corpus — every other MISMATCH is portable
(see below). The same `cortex_m::transpose` kernel also **rejects rank ≥ 5** (`op_transpose.cpp:44
… got 5/7`), which is why `pixel_shuffle` (389) / `pixel_unshuffle` (368) / rank-5 transpose SKIP.

### Kernel-guard / lowering SKIPs (cortex-m-attributed; reasons in [skips.md](skips.md))
- `cortex_m::transpose` rank ≤ 4 limit → 843 SKIPs across pixel_shuffle/unshuffle/permute/transpose.
- `cortex_m::minimum` (34) / `cortex_m::quantized_mul` (10) **`ET_CHECK`-abort** when the pass routes
  **non-int8** operands (`int32` / `float32`) into the int8 CMSIS kernel — a lowering bug: the rewrite
  fired without ensuring both operands are int8.

## Per-operator table (top non-OK; full ledger in [_work/per_op_table.txt](_work/per_op_table.txt))
Each row is one operator over its `k` samples. "CRASH" columns for `linear`/`_fft_r2c` are build-infra.

| operator | k | OK | MISM | CRASH | SKIP | TO |
|----------|--:|---:|----:|-----:|----:|---:|
| `_fft_r2c` | 442 | 0 | 0 | 23 | 0 | 419 |
| `stack` | 430 | 0 | 0 | 0 | 430 | 0 |
| `pixel_shuffle` | 760 | 371 | 0 | 0 | 389 | 0 |
| `pixel_unshuffle` | 741 | 373 | 0 | 0 | 368 | 0 |
| `zeros.out` | 345 | 0 | 0 | 0 | 345 | 0 |
| `ones.out` | 339 | 0 | 0 | 0 | 339 | 0 |
| `full.out` | 333 | 0 | 0 | 0 | 333 | 0 |
| `arange.start_out` | 322 | 0 | 0 | 0 | 322 | 0 |
| `arange.out` | 304 | 0 | 0 | 0 | 304 | 0 |
| `_pdist_forward` | 342 | 60 | 0 | 0 | 282 | 0 |
| `native_group_norm` | 278 | 42 | 0 | 0 | 236 | 0 |
| `index_put` | 435 | 218 | 217 | 0 | 0 | 0 |
| `fill.Scalar` | 348 | 164 | 184 | 0 | 0 | 0 |
| `where.self` | 367 | 185 | 30 | 0 | 152 | 0 |
| `copy` | 341 | 168 | 0 | 0 | 173 | 0 |
| `bitwise_left_shift.Tensor_out` | 390 | 219 | 171 | 0 | 0 | 0 |
| `expand_copy` | 340 | 185 | 0 | 0 | 155 | 0 |
| `bitwise_right_shift.Tensor_Scalar` | 442 | 288 | 154 | 0 | 0 | 0 |
| `max_pool2d_with_indices_backward.grad_input` | 129 | 11 | 117 | 0 | 1 | 0 |
| `linear.out` | 116 | 0 | 0 | 116 | 0 | 0 |
| `scatter_add.out` | 120 | 5 | 0 | 0 | 115 | 0 |
| `linear` | 114 | 0 | 0 | 114 | 0 | 0 |
| `topk.values` | 391 | 294 | 0 | 0 | 97 | 0 |
| `remainder.Tensor_out` | 227 | 136 | 91 | 0 | 0 | 0 |
| `scatter.src_out` | 85 | 0 | 0 | 0 | 85 | 0 |
| `remainder.Scalar` | 227 | 143 | 84 | 0 | 0 | 0 |
| `bitwise_left_shift.Tensor_Scalar` | 440 | 369 | 71 | 0 | 0 | 0 |
| `permute_copy` | 338 | 269 | 8 | 0 | 61 | 0 |
| `min.dim_min` | 407 | 339 | 0 | 0 | 68 | 0 |
| `bitwise_right_shift.Tensor_out` | 393 | 335 | 58 | 0 | 0 | 0 |

## Non-OK bucketed by "whose kernel ran" (filter 3a; 6,446 classified)
| bucket | MISMATCH | SKIP | CRASH | TIMEOUT | reading |
|--------|---------:|-----:|------:|--------:|---------|
| **CM-COMPUTE** | **8** | 869 | 0 | 0 | in-scope: 8 = the permute→transpose bug; SKIP = transpose rank / minimum·mul int8-only |
| **CM-QUANT-ONLY** | 810 | 1,965 | 70 | 402 | compute ran portable; boundary cortex_m only → portable/coverage, not cortex-m |
| **NO-CM** | 618 | 1,529 | 175 | 0 | no cortex_m op → portable/reference (incl. bitwise-shift UB); CRASH=build-infra |

## Ruled out (see [ruled_out/](ruled_out/))
1,428 of 1,436 MISMATCH are **portable-compute** divergences (`index_put` 211, `fill.Scalar` 180,
`max_pool2d_bwd` 103, `where`, `div`, `prod`, `logit`, …: CM-QUANT-ONLY) or **integer bitwise-shift
UB** (`bitwise_left/right_shift`, `remainder`, `fmod`: NO-CM, negative/oversized shift counts). Filter
3a removes them from cortex-m scope; the shift class is additionally filter-3d UB. The cortex_m
quantize/dequantize boundary is exonerated (55,128 OK graphs share it; mismatches concentrate in
specific portable ops).

## Coverage ledger (step 2 gate; full in [_work/coverage_ledger.txt](_work/coverage_ledger.txt))
- **231** scheduled ops (`EXECUTORCH_OPS`) → **203** produced in the corpus → **28** never lowered
  (seed-infeasible / partition attrition: `_fft_*`, `_native_batch_norm_legit`, `nonzero`,
  `masked_select`, `cat.out`, `split_*`, …). Assert holds: 203 + 28 = 231.
- **26** produced ops are cortex_m-compute-capable by the static map; in practice only
  transpose/minimum/quantized_mul/pad fire for single-op graphs.
- **No device coverage** for `linear`, `linear.out`, `_fft_r2c` (100 % runner-build attrition).
- Generation-time attrition: `max_pool2d_with_indices_backward` (×294) and `_native_batch_norm_legit`
  (×2) crashed at `build_job` time (in `corpus_v3/cortex-m/_crashes/`).

## Reproducing
A broker + at least one FVP cortex-m client must be running (the fleet this corpus was fed through):
```
bash _work/start_broker.sh          # broker on 15664/15665/15666
bash _work/start_fleet.sh 12 0      # 12 FVP cortex-m clients (--target cortex-m55)
python bugs/repro_permute-copy-transpose.py
```
Runner builds are cached per op-set under `tmp/fvp_runner_cache/` (see the fvp_runner.sh note below).

## Notes on the harness
- cortex-m's diverse portable op-sets force a fresh cmake+link **per job** (~0.7 job/s, ~24 h) —
  unlike ethos-u, whose NPU-delegated graphs all probe to the same near-empty op-set (one cached
  runner). For **this single-op run** an op-set ELF cache in `fvp_client/fvp_runner.sh` cut that to
  ~17 job/s, but it was **reverted afterward**: on multi-node graphs op-sets are nearly unique per
  graph, so the cache almost never hits and only balloons disk. `fvp_runner.sh` is back to
  mktemp-per-job; re-apply the cache locally for single-op sweeps only.
- All analysis tooling is in [_work/](_work/): `cm_kernels.py`/`cm_classify.py` (filter 3a),
  `cm_repro.py` (faithful device repro from stored `.pte`+inputs+reference), `cm_determinism.py`
  (3b gate), `analyze_table.py`, `coverage.py`, `cm_skips_report.py`.
