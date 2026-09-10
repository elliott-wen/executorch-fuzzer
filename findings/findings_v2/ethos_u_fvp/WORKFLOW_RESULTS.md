# Ethos-U55 differential-fuzzing results (Arm Corstone-300 FVP, int8 / Vela)

Backend **ethos-u**, quantized int8, lowered with Vela, executed on the **Corstone-300 FVP**
(Cortex-M55 + Ethos-U55) via the `fvp_client` worker fleet. The oracle is the **PT2E-quantized
CPU reference** (the `convert_pt2e` graph run on CPU — the *same int8 quantization space* as the
device), so a surviving diff is a genuine Vela/NPU compiler-or-kernel bug, **not** quantization
error (see `gen/export/job.py:121` — for quantized jobs the fp32 eager oracle is replaced by the
quantized reference). Every number below is device-verified on the FVP.

## 1. Corpus feed (58 701 jobs)

| status | count | share | meaning |
|---|---:|---:|---|
| ok | 40 255 | 68.6% | device matches the quantized reference |
| **mismatch** | **4 070** | 6.9% | device diverges from the quantized reference |
| skip | 14 364 | 24.5% | graph can't run on the target (see §5) |
| crash / timeout | **0** | 0% | no infra-induced failures across ~11.5 h |

Zero crashes/timeouts: the FVP build/execute path is clean (the infra rebuild — shared SDK,
`FETCH_ETHOS_U_CONTENT=OFF`, cmsis-nn guard, ccache — is documented in `analysis.md`).

## 2. Bisection of every mismatch (4 070 → 4 057 verdicts)

Each mismatch was bisected independently (output-set ddmin, keeping the target's cone) —
**not sampled**. 13 jobs segfault Vela during host lowering and are unclassified (0.3%).

| verdict | count | what it means |
|---|---:|---|
| OPERATOR | 3 078 | the target's minimal cone still diverges (candidate operator bug) |
| GRAPHOPT | 225 | diverges only when a co-returned sibling output is present |
| LOWERFAIL | 470 | the reduced sub-graph won't lower to Ethos-U (strict partitioning) |
| NOREPRO | 284 | divergence doesn't reproduce (near-tolerance / non-deterministic) |

The 3 078 OPERATOR verdicts span **190 distinct operators**.

## 3. Phase-2 single-op isolation — the key result

Each of the 190 OPERATOR ops was re-checked by **rebuilding just that op as a one-op graph fed
its exact runtime quantized inputs**, run on the FVP, diffed vs the quantized reference. This
separates genuine kernels from graph-context effects. **Unlike Vulkan (49 genuine kernel bugs),
Ethos-U's divergences are dominated by graph-context, not per-op kernels:**

| isolation class (per op) | count | interpretation |
|---|---:|---|
| GENUINE / non-finite (1st pass) | 26 | op diverged alone once — **candidate**, not yet confirmed |
| **CLEAN** | 122 | op **reproduces the reference alone** → divergence is graph-context (§4) |
| LOWERFAIL | 36 | op won't partition to the NPU alone → can't isolate (strict partitioning) |
| UNRESOLVED | 4 | isolation build/run wouldn't complete |

**A single GENUINE isolation is not a confirmed bug on int8.** Re-running each of the 26 candidates
**5×** (the gate the Vulkan run used, which I initially skipped) collapses the list — see
`REVERIFICATION.md`:

| after 5× re-verification | count | ops |
|---|---:|---|
| **STABLE (confirmed bug)** | **5** | `floor_divide.default`, `pow.Scalar_out`, `prod.default`, `prod.int_out`, `slice_scatter.default` |
| **INTERMITTENT (non-deterministic kernel)** | 8 | `bitwise_left_shift` (3/5), `bitwise_right_shift.Tensor_out` (2/5), `logit` (2/5), `mean.default`, `rsub.Scalar`, `clamp`, `fill.Scalar`, `_upsample_bilinear2d_aa` (1/5) |
| **FLAKY — dropped (near-tolerance noise, not a bug)** | 13 | `_adaptive_avg_pool2d`, `addmm`, `alias_copy`, `any.dims_out`, `div.Scalar`, `elu`, `eq.Tensor_out`, `mean.out`, `floor_divide.out`, `native_group_norm`, `native_layer_norm`, `bitwise_right_shift.Tensor_Scalar`(+`_out`) |

**Headline (corrected):** only **~5 operators are stably-reproducible single-op kernel bugs**; the
first-pass 26 was inflated ~5× by single-run flukes. Both surviving **dtype**-divergence ops are only
1/5 → the "device returns int where reference is float" signal is largely an **isolation-harness
artifact**, not a device bug. The 122 CLEAN ops still diverge only in graph context (§4).

## 4. Graph-optimization bugs — pinpointed mechanism — `GRAPH_CONTEXT.md`
122 ops are CLEAN in isolation but diverge in-graph, and the bisection flagged 225 GRAPHOPT
(sibling-dependent) cases. A **per-side A/B on device** (target's cone alone vs +sibling) pinpoints
what's actually happening, after removing two confounds:
- the **2^63 "garbage" deltas are INT64 sentinels in the *reference*** (cumsum-type ops on the CPU
  int64 path), not device aliasing — they mismatch *alone* → **CONFOUND**, not graph-opt;
- **57% of testable GRAPHOPT cases are CONFOUND** (mismatch alone / flaky), only **~43% are genuine
  sibling-dependent** graph-opt.

For the genuine cases the deep-dive **names the mechanism**: the dominant one is a **requantization
scale error** — device ≈ reference × a wrong scale (`gelu` ×−3.79, `logit` ×0.108, `logit` ×−1.5e18):
co-returning a sibling changes Vela's partition and `(scale, zero_point)` choice, so the target is
dequantized with the wrong scale. Secondary: **dropped store** (`view_copy`, `pixel_unshuffle` →
all-zero output). This is a **Vela quantization-parameter / partitioning bug**, not buffer-aliasing
(QNN's bug) — same corpus, different backend fault.

## 5. Skip bugs (24.5%) — coverage gaps of the target
Skips are graphs the pipeline lowers but the device rejects at load/run time — overwhelmingly
**ops with no bare-metal portable kernel** (e.g. `aten::_fft_r2c.out` → "Missing operator", runner
load fails). These are *target coverage gaps*, not compiler bugs; enumerated in `skip_log.tsv`.

## 6. Cross-device differential — `CROSS_DEVICE_DIFF.md`
Against the confirmed Vulkan (49) and QNN (32) operator bugs: `bitwise_left_shift`, `prod`, and
`rsub.Scalar` diverge on **all three** backends; 5 more Ethos-U bugs are shared with QNN; **16 are
Ethos-unique** (reductions `mean`/`prod`, `native_group_norm`/`native_layer_norm`, `slice_scatter`,
`_adaptive_avg_pool2d` — int8/Vela specific). Ethos-U's *profile* is distinct: kernel bugs are few;
the mass is graph-context/quant.

## Artifacts
- `bisect_results.tsv` — every mismatch's bisection verdict.
- `skip_log.tsv` — all mismatch + skip rows (status, job, reason, op-chain).
- `bugs/operators/` — one device-verified `.md` per confirmed operator bug + `INDEX.md`.
- `GRAPH_CONTEXT.md`, `CROSS_DEVICE_DIFF.md` — the two analyses above.
- `operator_isolation.tsv` — per-op isolation class (GENUINE/CLEAN/LOWERFAIL/…).
