# Consistency-divergence categories — per backend (single-operator corpus)

Four-category classification of every **device-verified, backend-attributed MISMATCH** in the
single-operator corpus runs. Scope matches the paper's consistency table: only graphs that
**deployed and executed** are counted (CRASH / HANG / runtime-SKIP are excluded — see
`RUNTIME_SKIP_TAXONOMY.md`).

## Counting basis

A unit is a **distinct operator overload** (`sub.out` and `sub.Scalar` count separately, as in the
per-run tables) that is:

1. **delegated** — `delegated.ops ≥ 1`, so the backend's kernel actually ran (filter 3a of
   `analysis_single.md`). Portable-fallback mismatches are excluded; this is a large correction on
   some backends (NXP: 3,289 raw mismatches → 38 delegated).
2. **determinism-gated** — reproduces on all N≥5 re-runs of the same `.pte`. INTERMITTENT and flaky
   rows are excluded.
3. **mechanism-assigned** — counted once, in its *dominant* documented mechanism by record count.

Categories: **NumSem** (numerical-semantics: non-finite handling in either direction, signed-integer
semantics, integer overflow/accumulation, shift UB, precision) · **Comp** (structurally valid output,
value error not explainable by convention or precision; includes operand-ignored-with-nonzero-output)
· **Map** (values not mapped to the output: wrong layout/indexing, dropped store, foreign-buffer
read/replication) · **Meta** (shape or dtype of the returned tensor).

Tie-break between Comp and Map when the output is input-independent: **zero-initialised buffer → Map;
non-zero but invariant to the operand → Comp.**

Where a backend ran on several devices, the row is the **device with the larger confirmed operator
set**, named in the row; the footnotes give what the other device adds.

## Table

| backend | device reported | NumSem | Comp | Map | Meta | total ops | root causes |
|---|---|--:|--:|--:|--:|--:|--:|
| QNN (HTP) | Snapdragon phone | **76** | 18 | 9 | 0 | **103** | 6 |
| Samsung ENN | Exynos E9965 | 12 | 10 | **18** | 0 | **40** | ~4 |
| CoreML | Apple-Silicon Mac | **16** | 8 | 2 | 0 | **26** | 8 |
| OpenVINO | Intel CPU host | 6 | 8 | 8 | **3** | **25** | 7 |
| Vulkan | Moto phone | 2 | 0 | **19** | 0 | **21** | 6 |
| MediaTek Neuron | MTK phone | 0 | 1 | 5 | **14** | **20** | 6 |
| VGF (TOSA→Vulkan) | lavapipe + ML-SDK emu | **10** | 5 | 1 | 0 | **16** | 3 |
| CUDA (AOTI) | H200 | **11** | 0 | 0 | 0 | **11** | 3 |
| Ethos-U | Corstone-300 FVP | 3 | 2 | 2 | 0 | **7** | 7 |
| XNNPACK | x86-64 host | **4** | 0 | 0 | 0 | **4** | 3 |
| Cortex-M | Corstone-300 FVP | 0 | 0 | **1** | 0 | **1** | 1 |
| NXP Neutron | eIQ NSYS simulator | **1** | 0 | 0 | 0 | **1** | 1 |
| Cadence (Xtensa HiFi) | host generic reference kernels | 0 | **1** | 0 | 0 | **1** | 1 |
| *portable (reference)* | *x86-64 host* | *8* | *1* | *1* | *0* | *10* | *9* |

`root causes` = distinct diagnosed mechanisms behind those operators. **Read it next to the operator
count**: op counts measure blast radius, not the number of bugs. QNN's 76 NumSem operators are one
defect (the HTP fp16 path implements no IEEE non-finite semantics); MediaTek's 14 Meta operators are
one defect (int32 emitted for int64 outputs); 14 of Vulkan-Moto's 19 Map operators are one driver
corruption bug.

## Per-cell composition

**QNN — Snapdragon phone** (103 ops, from `qnn_phone/per_operator.md`; `fp16-prec` and `int64`
columns ruled out). NumSem 76: the non-finite class (`logit`, `sqrt`, `log`, `exp`, `cos`, `sin`,
`atan`, `mm`, `linear`, `elu`, `gelu`, `_softmax`, … — `inf → ±131008/±65472/~0`, `nan → wrong
finite`) plus integer semantics (`remainder`, `fmod`, `floor_divide`, `prod.int_out`,
`bitwise_*_shift`, `isnan`). Comp 18: `_native_batch_norm_legit.no_stats` (`[-1,1] → [-15256,
35104]`), `rsub.Scalar`/`sub.out` (`18→9`), the six `bitwise_and/or/xor` overloads (return `1`
instead of the bitwise result), comparison/reduction ops. Map 9: `replication_pad2d/3d.out` (→ all
zeros), `index.Tensor_out`/`index_select(.out)` (garbage tail `-129728`), `copy`, `roll`,
`select_scatter`, `flip`.

**Samsung ENN — Exynos E9965** (40 CONFIRMED rows in `samsung_e9965/README.md`). Map 18: the integer
data-movement family — garbage read (`permute_copy`, `roll`, `t_copy`, `transpose_copy.int`,
`select_copy.int`) and dropped store (`constant_pad_nd`, `expand_copy`, `squeeze_copy.dim/.dims`,
`sub.Scalar`, `mul.Scalar`, `_softmax.out`, `embedding`, `pixel_shuffle`, `pixel_unshuffle`,
`view_copy`, `unsqueeze_copy`, `select_scatter`). NumSem 12: `div.Scalar` (fp16 overflow from a
finite input), `gelu.out`, `sqrt.out`, `rsqrt.out`, `logit`, `clamp.out`, `var.correction_out`,
`linear.out`, `prod.int_out`/`sum.IntList_out` (uint8 saturation), `remainder.Tensor_out`,
`bitwise_left_shift.Tensor_out`. Comp 10: `rsub.Scalar`, `add.Scalar`, `sub.out`, `hardtanh`,
`maximum.out`, `glu.out`, `bmm.out`, `relu`, `avg_pool2d.out`, `linear`.

**CoreML — Apple-Silicon Mac** (26 delegated-mismatch ops). NumSem 16: precision reductions
(`var.correction`, `var.correction_out`, `_native_batch_norm_legit.no_stats`, `logit`,
`pow.Scalar_out`, `remainder.Tensor_out`, `convolution`), `gelu.out`, `floor_divide(.out)`,
`min/max.unary_out`, `bitwise_left_shift.Tensor_out` (→ `-2³¹`), `prod.int_out`, `sum.IntList_out`,
`mean` (integer-truncated mean of an int input). Comp 8: `atan2.out` (`atan2(0,x<0) → 0` instead of
`π`), `avg_pool2d.out` (~2× divisor), `_upsample_bilinear2d_aa.out` (antialias operand ignored),
`pow.Tensor_Tensor_out`, `logical_xor.out`, `any.all_out`, `any.dims_out`, `mean.out`. Map 2:
`remainder.Scalar` (all-zero output), `max_pool2d_with_indices_backward.grad_input` (wrong scatter).
*Uses `coreml_mac` (single-op); `coreml_mac2` is the multi-op run.*

**OpenVINO — Intel CPU host** (25 ops). Map 8: dropped store on `clone`, `alias_copy`,
`lift_fresh_copy`, `t_copy`, `transpose_copy.int`, `rsub.Scalar`; layout/indexing on
`diagonal_copy` (returns row 0) and `unfold_copy` (same multiset, row-major order). Comp 8:
`reflection_pad1d.out`, `reflection_pad2d(.out)`, `remainder.Tensor_out`, `remainder.Scalar_out`,
`pow.Scalar_out`, `var.correction`, `var_mean.correction`. NumSem 6:
`_native_batch_norm_legit.no_stats` (finite leaf and finite reference → `nan`), plus fp32
accumulation on `addmm.out`, `bmm.out`, `convolution`, `linear`, `linear.out`. Meta 3:
`min.dim_min`, `max.dim_max`, `topk.values` (`out=` tensors never resized). Three
lower-confidence UB rows (`bitwise_left_shift.*`, `prod.int_out`) are excluded.

**Vulkan — Moto phone** (21 ops). Map 19: shared backend bugs `clamp.Tensor` (in-range → 0),
`index_put` (misplaced writes), `copy` (unrelated live buffer), `full.out`, `fill.Scalar` (constant
buffer aliased → 0); plus the **Moto-driver-unique** copy/view corruption that collapses the output
to a repeated element — `bmm`, `expand_copy`, `permute_copy`, `constant_pad_nd`, `lift_fresh_copy`,
`pixel_shuffle`, `clone`, `slice_copy`, `squeeze_copy.dim/.dims`, `alias_copy`, `view_copy`, `abs`,
`logit` (2,521 jobs that are OK on ASUS **and** Samsung). NumSem 2: `floor_divide`, `mean`.

**MediaTek Neuron — MTK phone** (20 mismatch ops; `copy` is CRASH-only and excluded). Meta 14: one
int32-for-int64 defect across `fill.Scalar`, `full.out`, `arange(.start)_out`,
`squeeze_copy.dim/.dims`, `expand_copy`, `view_copy`, `permute_copy`, `t_copy`, `constant_pad_nd`,
`pixel_shuffle`, `pixel_unshuffle`, `pow.Tensor_Scalar_out` — the dtype error becomes a value error
because two int32 lanes are read back as one int64 (`fill.Scalar(1) → [4294967297, 4294967297, 0,
0]`). Map 5: `roll` (wrong cyclic order), `convolution` (wrong output layout),
`replication_pad{1,2,3}d.out` (wrong ordering). Comp 1: `native_group_norm`.

**VGF** (16 ops, `vgf_single`). NumSem 10: fp16 overflow from finite leaves on `leaky_relu.out`,
`gelu.out`, `clamp.out`, `relu`, `elu.out`, `floor_divide(.out)`, plus
`bitwise_left_shift.Tensor_out`, `sum.IntList_out`, `prod.int_out`. Comp 5: `native_group_norm`,
`native_layer_norm`, `slice_scatter`, `where.self`, `where.self_out`. Map 1: `fill.Scalar` (all zeros
regardless of the scalar, 356 cases, fully delegated). *The `where.self` IO-width defect
("Output tensor byte size 4 ≠ IO allocation 2") is a Meta-class bug but surfaces as a SKIP, so it is
not in this table.*

**CUDA (AOTI) — H200** (11 ops, all NumSem): `bitwise_left/right_shift.Tensor_Scalar(.out)` (shift ≥
bitwidth / negative — UB, ~570 cases), `prod.int_out` and `sum.IntList_out` (integer accumulation:
uint8 product wraps to 0 on CPU, returns 255 on device), and fp16 non-finite disagreement on
`gelu.out`, `floor_divide(.out)`, `grid_sampler_2d`, `elu.out` (`NaN` vs `Inf`).

**Ethos-U — Corstone-300 FVP** (7 ops). NumSem 3: `floor_divide` (rounding off-by-one at the integer
boundary), `bitwise_right_shift.Tensor_out` (logical instead of arithmetic on negatives),
`remainder.Tensor_out` (`1 % -3`: `-2` → `1`). Comp 2: `fill.Scalar` (fill value never applied;
output is a fixed input-derived pattern, identical for `value=1` and `value=3` — it also downcasts
int64→int32, a secondary Meta effect), `sub.out`. Map 2: `rsub.Scalar` (collapses to ~0 past the
first elements), `clamp.Tensor_out` (tensor-bound broadcast fails → first value replicated).

**XNNPACK — x86-64 host** (4 ops, all NumSem): `sqrt` (`sqrt(neg) → -0.0`, `sqrt(inf) → NaN`),
`gelu` (`gelu(inf)`: `NaN → inf`), `relu`/`hardtanh` (`NaN` → clamp bound via fp16 SIMD
fmax/fmin laundering).

**Cortex-M — Corstone-300 FVP** (1 op, Map): `permute_copy` — a general/identity permute is lowered
to `cortex_m::transpose`, a fixed axis-swap, so the output is the transpose rather than the
permutation. int8-only, so the NumSem non-finite class cannot exist here.

**NXP Neutron — eIQ NSYS simulator** (1 op, NumSem): int8 `relu` returns `0` at the `+127`
saturation boundary (38/38, `max|Δ| = 127.0` exactly) — the top int8 code overflows to `-128` and
relu clamps to the zero point. The other 193 `relu` graphs and all 190 `abs` graphs are correct, so
the surrounding quantize/dequantize pair is sound. Only 2 operators ever reach this delegate, so the
row is a two-operator surface, not a clean bill of health.

**Cadence (Xtensa HiFi) — host generic reference kernels** (1 op, Comp; from
`cadence_hostsim/README.md`). `addmm.out` → `cadence::quantized_linear`: the quantizer's
`AddmmPattern` claims every `aten.addmm` and never reads `beta`/`alpha`, so the fused
`quantized_linear` (which has no scalar multipliers) computes `bias + mat1@mat2` regardless.
227/228 graphs reaching that kernel fail, all deterministic 5/5, and **the single passing graph is
the only one in the corpus with `beta=1, alpha=1`**. Output is structurally valid and non-zero but
invariant to the two scalars → Comp by the tie-break. The parallel unquantized pass
`ReplaceAddMMWithLinearPass` does guard on `beta == 1.0 and alpha == 1.0`; Cadence runs
`quant=ALWAYS`, so traffic takes the unguarded path.

Cadence needs the **pass-based filter-3a substitute** (as `cortex_m` does): `delegated.ops == 0` on
all 84,581 jobs, so the gate is "does the `.pte` call a `cadence::` kernel" — true for only 1,066
graphs (1.26%). 422 of the run's 649 mismatches and all 10 crashes are portable-fallback and
excluded. Two Cadence defects are excluded by scope but are larger than the row: **356 graphs (33.4%
of all cadence-kernel graphs) cannot load at all** because `cadence::{transposed_convolution,
avg_pool2d, conv1d}.out` are emitted by the AOT passes but absent from `functions.yaml`; and 991
graphs return a **different output dtype than eager** (`masked_fill`/`fill`/`tril`/`mul.Scalar`),
a Meta-class divergence the harness currently records as SKIP because it cannot decode the buffer.

**portable (reference kernels)** — not a delegate; included for comparison. NumSem 8: `gelu`
(`NaN → inf`), `_log_softmax` (`NaN → 0`), `grid_sampler_2d` (`inf → NaN`), `native_group_norm`,
`addmm` (finite → all-NaN), `remainder`, `fmod` (sign), `prod` int (`0 → 1`). Comp 1:
`_native_batch_norm_legit.no_stats` (Δ≈4). Map 1: `max_pool2d_with_indices_backward`.

## Device footnotes

- **Vulkan.** Moto (5,821 mismatches / 21 confirmed ops) is reported. ASUS (3,704) and Samsung
  (3,634) share the 5 Map bugs but get the Moto copy/view corruption **right**, and add **5 NumSem
  operators Moto handles correctly** (`gelu`, `sqrt`, `rsqrt`, `pow`, `mean` — non-finite saturation).
  Neither device's row is a superset: the union is 26 operators. Non-finite handling is a *driver*
  property here, not a backend property.
- **XNNPACK.** x86-64 (4 ops) is reported. arm64 adds **2 operators x64 gets right** — `log` and
  `logit` both saturate `+inf` to `88.375` (≈ `ln(FLT_MAX)`) in the NEON `log` polynomial — and
  shares `gelu`; it does **not** reproduce the `sqrt` bug. Union = 6 operators. Four more x64 NumSem
  operators (`exp`, `rsqrt`, `minimum`, `maximum`) are confirmed by hand-fed triggers but are not in
  the count because the corpus under-samples the exact input values.
- **QNN.** Phone (103 ops) is reported over the x86 HTP emulator (same operator profile, additionally
  gives real crash signals) and the two short Samsung runs (`qnn_samsung` 977, `qnn_samsung_sm8750`
  1,237 mismatches).
- **CoreML** uses the single-op run (`coreml_mac`); **VGF** uses `vgf_single`. The `coreml_mac2` and
  `vgf` (corpus_v4) runs are multi-op and their headline findings are graph-optimization
  dropped-stores, which a single-operator corpus cannot surface by construction.

## What this table excludes

- **Crashes, hangs, runtime rejections.** Excluded by scope, but they dominate on some backends: NXP
  2,182 crashes vs 38 delegated mismatches, CUDA 794 skips, CoreML 268 crashes + 26 timeouts,
  Vulkan ~555 real aborts, VGF 10,358 runtime skips.
- **Tolerance-scale divergence** below `rtol=0.01, atol=0.001`, and QNN's ~250 fp16-vs-fp32 deltas.
- **Nondeterministic rows**, removed by the N≥5 gate (OpenVINO 2/237 intermittent; Samsung 9
  INTERMITTENT operators; VGF `acos`/`addmm`/`div`/`logit`/`sinh`/`_softmax` at 0/5). A single pass
  over-reports int8 operator bugs several-fold.
- **`bitwise_*_shift` where it is an oracle artifact** (int64 `2^63` reference sentinels): ruled out
  on most backends, retained only where device-confirmed (CUDA, Ethos-U, CoreML, VGF, Samsung).
