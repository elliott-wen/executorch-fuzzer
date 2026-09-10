# Runtime MISMATCH — per backend/device

A **MISMATCH** is a numeric divergence between the device output and the eager ATen reference on a
graph that exported, lowered and executed successfully. Counts are the per-run feeder verdicts;
`ops` is the number of distinct operators involved; `non-finite` is the subset whose divergence is a
NaN/inf position difference rather than a finite value difference.

**Two filters must be applied before these numbers mean anything.**

1. **`bitwise_*_shift` is ruled out**, though it tops nearly every run at 200–340 records. Portable's
   analysis attributed it to int64 `2^63` reference sentinels plus shift-UB in the oracle. Only CUDA
   and Ethos-U have a *confirmed* shift bug.
2. **Portable-fallback rows are not backend findings.** Only graphs with `delegated.ops ≥ 1` can be
   attributed to the backend. The effect is large: NXP reports 3,289 mismatches of which **38** are
   on its delegate.

## Table

| backend / device | MISMATCH | ops | non-finite | confirmed backend-attributed operators | dominant mechanism |
|---|--:|--:|--:|---|---|
| qnn_phone | 8,899 | 118 | 6,247 (70%) | `rsub`/`sub` (`18→9`, then garbage), `_native_batch_norm` (`[-1,1]→[-15256,35104]`), `replication_pad2d/3d` (edge → all 0), `bitwise_or`/`xor` (`255→1`), `index_select` (garbage tail `-129728`) | non-finite dominated; scalar-arith + indexing corruption |
| qnn_emulator | 7,699 | 113 | 5,505 (72%) | same set as the phone (`rsub` value bug, non-finite mishandling) | same as phone; emulator additionally gives real crash signals |
| openvino | 6,000 | — | — | `clone` 432, `lift_fresh_copy` 426, `alias_copy` 419, `t_copy` 348 (all **ZEROED**) · `reflection_pad1d` 385, `reflection_pad2d` 352, `diagonal_copy` 327 (wrong value) | **ZEROED across the whole copy/view/alias family** |
| vulkan_moto | 5,821 | 83 | 1,046 | shared with ASUS: `clamp.Tensor` 308, `index_put` 213, `copy` 116, `full`/`fill` 42 · **Moto-only 2,521**: `bmm`, `expand_copy`, `permute_copy`, `clone`, `alias_copy`, `slice_copy`, `view_copy`, `squeeze_copy`, `pixel_shuffle`, `constant_pad_nd` | driver-specific corruption of the copy/view family |
| samsung_e9965 | 4,854 | 71 | 1,904 | `_softmax.out` 444 (Z), `gelu.out` 440, `rsub.Scalar` 439, `sub.Scalar` 403 (Z), `add.Scalar` 391, `sub.out` 379, `div.Scalar` 305 (NF), `mul.Scalar` 204 (Z) | scalar-broadcast arithmetic: mixed ZEROED / wrong-value |
| vulkan_asus | 3,704 | 45 | 1,265 | `clamp.Tensor` 308 (in-range → 0), `index_put` 213 (misplaced writes), `copy` 116 (→ unrelated buffer), `full`/`fill` 42 (→ 0) | value / index / constant-buffer bugs |
| vulkan_samsung | 3,634 | 43 | 1,191 | same four as ASUS (0.97 similarity with ASUS) | same as ASUS — Moto is the outlier, not Samsung |
| ethos_u | 3,402 | 55 | 603 | `fill.Scalar` (scalar ignored), `rsub.Scalar` (collapses to 0 past element 1), `floor_divide` (off-by-one at integer boundary), `bitwise_right_shift` (logical not arithmetic), `clamp.Tensor` (broadcast → constant), `sub.out`, `remainder.Tensor` (wrong sign) | 7 individually diagnosed value bugs |
| nxp_imxrt700 | 3,289 | 55 | 92 | **`relu` 38 only** — ZEROED at int8 max (`127→0`) | rest is portable fallback: the quantizer's patterns match `.Tensor` overloads while the corpus emits `.Scalar`, so nothing gets delegated |
| vgf | 3,149 → **2,335 delegated** | 33 → 25 | 1,675 | `native_group_norm` 533, `leaky_relu.out` 411 (NF), `fill.Scalar` 356 (**all zeros regardless of the scalar**), `gelu.out` 350 (NF), `floor_divide.out` 126 | 68% non-finite (fp16/TOSA saturation); `fill.Scalar` is the cleanest value bug |
| xnnpack_x64 | 2,358 | 26 | 1,117 | `sqrt` 353 (`sqrt(neg)=-0.0`, `sqrt(inf)=NaN`) · `gelu` (inf→inf) | almost entirely non-finite saturation |
| xnnpack_arm64 | ~2,383 | 29 | 988 | `logit` 187, `log` 180 (saturate to ~88.38) · `gelu` (inf→inf, shared with x64) | same backend, **different operators** from x64 |
| mtk_phone | 1,945 → **2,287 corrected** | 44 | 388 | **int64 outputs corrupted** across `fill.Scalar`, `full.out`, `arange`, `squeeze_copy`, `expand_copy`, `view_copy`, `permute_copy`, `t_copy` · `roll` 187 (wrong cyclic order), `native_group_norm` 97, `convolution` 9 (wrong layout), `replication_pad*` | one dtype-wide int64 defect + a few value bugs |
| portable_x64 | 1,742 | 21 | 764 (46%) | `gelu` 369 (NaN→inf), `remainder`/`fmod` 247 (sign), `_log_softmax` 128 (NaN→0), `grid_sampler_2d` 102 (inf→NaN), `batch_norm` 44, `native_group_norm` 29, `prod` int 25 (0→1), `addmm` 23 (finite→all-NaN) | 983 confirmed / 759 ruled out |
| cuda_h200 | 1,504 | 23 | 472 | shift UB ~568 (shift ≥ bitwidth — **real here**), `prod.int_out` 20 / `sum.IntList_out` 17 (integer overflow), `gelu`/`floor_divide`/`grid_sampler_2d`/`elu` ~134 (fp16 threshold) | integer UB + fp16 non-finite threshold |
| cortex_m | 1,485 | 41 | **0** | `permute_copy` (identity permute lowered to a fixed axis-swap) · `index_put` 217, `fill.Scalar` 184 | int8-only, so no non-finite class exists |
| coreml_mac | 1,467 | 28 | — | `var.correction` 158, `gelu.out` 77, `_upsample_bilinear2d_aa` 77, `mean` 69, `atan2`, `avg_pool2d`, `mean-int` | reduction precision + interpolation |
| qnn_samsung_sm8750 | 1,237 | 62 | 710 | `logit` 78/72, `rsub.Scalar` 68, `gelu.out` 66, `mm.out` 59 | short run; same profile as qnn_phone |
| qnn_samsung | 977 | 57 | 531 | `logit` 58/53, `rsub.Scalar` 49, `gelu.out` 46, `mm.out` 45 | short run; same profile as qnn_phone |

(Z) = ZEROED · (NF) = non-finite. `—` = the run has no machine-readable per-row log; figures from its README.

## Three cross-cutting mechanisms

**1. Silent zeroing / dropped stores — the most dangerous class.** The output collapses to zero or to
a repeated element, with no crash and no error. It appears *independently* on openvino (~1,625
records across the copy/view/alias family), samsung (`_softmax`, `sub.Scalar`, `mul.Scalar`), vgf
(`fill.Scalar` 356 — the scalar operand is dropped entirely), ethos_u (`rsub.Scalar`), vulkan
(`clamp`→0, `full`→0) and nxp (`relu` `127→0`). Nothing signals it to the user.

**2. Non-finite handling is a device property, not a backend property.** It splits *within* a single
backend: XNNPACK's `sqrt` is buggy on x64 and clean on arm64, while `log`/`logit` are the reverse;
Vulkan's `gelu` non-finite bugs fire on ASUS but not Moto. This is why 46–72% of raw mismatch counts
are non-finite and cannot be attributed to a backend alone.

**3. The copy/view/alias family is the most-affected operator group overall.** `clone`, `alias_copy`,
`lift_fresh_copy`, `t_copy`, `view_copy`, `expand_copy`, `permute_copy`, `slice_copy` — nominally
no-op data movements — are the top cluster on openvino (ZEROED), vulkan_moto (corruption) and mtk
(int64 corruption). Operators that should be trivially correct are among the least reliable measured.

## Caveat for any single-device claim

The three Vulkan phones and the two XNNPACK hosts show that **cross-device differential testing
changes the conclusions**: 2,521 Moto-only mismatches would have been invisible on a single device,
and a later Samsung S26U run landed in the ASUS class (0.97 mismatch similarity), making Moto the
outlier. Any table listing a backend's mismatch operators without naming the device is wrong for
some other device.
