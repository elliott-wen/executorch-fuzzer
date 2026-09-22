# OpenVINO single-op — confirmed bug index

All entries are **device-verified** on the OpenVINO CPU host runtime, **delegated** (`delegated.ops≥1`,
so the OpenVINO delegate genuinely ran the op — not a portable fallback), and **determinism-gated
N=5** (235/237 sampled failing jobs reproduce on all 5 re-runs; 2 intermittent, both max_pool crash).

Reproduce any of them: `PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/bugs/repro.py`
(runs one representative job per mechanism), or pass explicit `job_id`s.

| # | mechanism | operators | doc |
|---|---|---|---|
| 1 | **ZEROED** (dropped store) | `clone`, `alias_copy`, `lift_fresh_copy`, `t_copy`, `transpose_copy.int` (+ partial: `reflection_pad1d.out`, `rsub.Scalar`, `var.correction*`) | [zeroed_copy_ops.md](zeroed_copy_ops.md) |
| 2 | **WRONG-VALUE** (view transform ignored) | `diagonal_copy`, `unfold_copy` | [wrongvalue_view_ops.md](wrongvalue_view_ops.md) |
| 3 | **WRONG-VALUE** (math) | `reflection_pad2d(.out)`, `reflection_pad1d.out`, `remainder.Tensor_out`, `remainder.Scalar_out`, `pow.Scalar_out`, `var_mean.correction`, `var.correction` | [wrongvalue_math_ops.md](wrongvalue_math_ops.md) |
| 4 | **NONFINITE** (device introduces nan, finite leaf+ref) | `_native_batch_norm_legit.no_stats` | [nonfinite_batchnorm.md](nonfinite_batchnorm.md) |
| 5 | **WRONG-SHAPE** (`out=` tensors not resized) | `min.dim_min`, `max.dim_max`, `topk.values` | [wrongshape_out_variants.md](wrongshape_out_variants.md) |
| 6 | **CRASH** (native abort) | `max_pool2d_with_indices_backward.grad_input` (+ rare `unfold_copy`, `reflection_pad2d.out`, `convolution`) | [crash_maxpool_backward.md](crash_maxpool_backward.md) |
| 7 | **VALUE** (fp32 accumulation, low severity) | `addmm.out`, `bmm.out`, `convolution`, `linear(.out)` | [value_fp_accumulation.md](value_fp_accumulation.md) |
| — | **VALUE / UB** (lower confidence) | `bitwise_left_shift.Tensor_out`, `bitwise_left_shift.Tensor_Scalar_out`, `prod.int_out` | [suspect_ub_dtype.md](suspect_ub_dtype.md) |

SKIP coverage gaps (OpenVINO delegate blob execute failures) and CRASH reason tables are in
[../skips.md](../skips.md). Ruled-out suspects (portable fallback, reference-nan) are in
[../ruled_out/](../ruled_out/).
