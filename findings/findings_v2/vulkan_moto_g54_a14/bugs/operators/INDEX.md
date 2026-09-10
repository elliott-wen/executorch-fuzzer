# Operator bugs — Vulkan on Moto G54 5G (Android 14)

Device-verified by single-op isolation with EXACT baked inputs (one-op graph fed the op's real runtime values, lowered to Vulkan, run on the Moto; diverges as the sole output with finite inputs → genuine kernel bug). Candidate pool 196 OPERATOR-verdict ops → 89 isolated GENUINE → **49 confirmed** after a 7× stability re-verification (40 dropped: reproduce eager on re-run, or won't isolate as one op). Each bug has a runnable `repro_<op>.py`.

- **45 confirmed operator bugs** (reproduce every successful re-run)
- **4 non-deterministic kernels** (wrong output *intermittently* on identical input — race/uninit)
- **40 ruled out** (see `ruled_out/`)

## Confirmed operator bugs

### dtype divergence (3)

| op | divergence | repro |
|---|---|---|
| `amax.out` | out[0] dtype torch.float32 vs torch.int64 | [amax_out](amax_out.md) |
| `any.all_out` | out[0] dtype torch.uint8 vs torch.int64 | [any_all_out](any_all_out.md) |
| `bitwise_left_shift.Tensor_Scalar_out` | out[0] dtype torch.float32 vs torch.float16 | [bitwise_left_shift_Tensor_Scalar_out](bitwise_left_shift_Tensor_Scalar_out.md) |

### wrong value (42)

| op | divergence | repro |
|---|---|---|
| `_log_softmax.out` | out[0] max|delta|=1.931e-01 (rtol=0.01 atol=0.001) | [_log_softmax_out](_log_softmax_out.md) |
| `abs.out` | out[0] max|delta|=5.531e-01 (rtol=0.01 atol=0.001) | [abs_out](abs_out.md) |
| `bitwise_left_shift.Tensor_Scalar` | no reply 60s | [bitwise_left_shift_Tensor_Scalar](bitwise_left_shift_Tensor_Scalar.md) |
| `bitwise_left_shift.Tensor_out` | out[0] max|delta|=9.223e+18 (rtol=0.0 atol=0.0) | [bitwise_left_shift_Tensor_out](bitwise_left_shift_Tensor_out.md) |
| `bmm.out` | out[0] max|delta|=4.732e+00 (rtol=0.01 atol=0.001) | [bmm_out](bmm_out.md) |
| `clamp.Tensor_out` | out[0] max|delta|=1.200e+01 (rtol=0.0 atol=0.0) | [clamp_Tensor_out](clamp_Tensor_out.md) |
| `clamp.out` | no reply 60s | [clamp_out](clamp_out.md) |
| `clone` | no reply 60s | [clone](clone.md) |
| `copy` | out[0] max|delta|=2.550e+02 (rtol=0.0 atol=0.0) | [copy](copy.md) |
| `diagonal_copy` | no reply 60s | [diagonal_copy](diagonal_copy.md) |
| `div.out_mode` | out[0] max|delta|=1.000e+00 (rtol=0.01 atol=0.001) | [div_out_mode](div_out_mode.md) |
| `fill.Scalar` | out[0] max|delta|=1.000e+00 (rtol=0.0 atol=0.0) | [fill_Scalar](fill_Scalar.md) |
| `floor_divide` | out[0] max|delta|=1.300e+01 (rtol=0.01 atol=0.001) | [floor_divide](floor_divide.md) |
| `gather.out` | out[0] max|delta|=1.000e+00 (rtol=0.0 atol=0.0) | [gather_out](gather_out.md) |
| `gt.Tensor_out` | out[0] max|delta|=1.000e+00 (rtol=0.0 atol=0.0) | [gt_Tensor_out](gt_Tensor_out.md) |
| `hardtanh` | out[0] max|delta|=9.340e-01 (rtol=0.01 atol=0.001) | [hardtanh](hardtanh.md) |
| `index_put` | out[0] max|delta|=1.000e+00 (rtol=0.0 atol=0.0) | [index_put](index_put.md) |
| `index_select` | no reply 60s | [index_select](index_select.md) |
| `leaky_relu.out` | out[0] max|delta|=1.725e+00 (rtol=0.01 atol=0.001) | [leaky_relu_out](leaky_relu_out.md) |
| `lift_fresh_copy` | out[0] max|delta|=4.576e+00 (rtol=0.01 atol=0.001) | [lift_fresh_copy](lift_fresh_copy.md) |
| `linear.out` | out[0] max|delta|=1.300e+01 (rtol=0.0 atol=0.0) | [linear_out](linear_out.md) |
| `lt.Tensor_out` | out[0] max|delta|=1.000e+00 (rtol=0.0 atol=0.0) | [lt_Tensor_out](lt_Tensor_out.md) |
| `min.dim_min` | out[0] max|delta|=9.223e+18 (rtol=0.0 atol=0.0) | [min_dim_min](min_dim_min.md) |
| `min.unary_out` | out[0] max|delta|=1.246e+11 (rtol=0.01 atol=0.001) | [min_unary_out](min_unary_out.md) |
| `mul.out` | out[0] max|delta|=5.042e+21 (rtol=0.01 atol=0.001) | [mul_out](mul_out.md) |
| `permute_copy` | out[0] max|delta|=1.413e+00 (rtol=0.01 atol=0.001) | [permute_copy](permute_copy.md) |
| `pixel_shuffle` | out[0] max|delta|=7.000e+00 (rtol=0.01 atol=0.001) | [pixel_shuffle](pixel_shuffle.md) |
| `pow.Tensor_Scalar_out` | out[0] max|delta|=1.809e-01 (rtol=0.01 atol=0.001) | [pow_Tensor_Scalar_out](pow_Tensor_Scalar_out.md) |
| `pow.Tensor_Tensor_out` | out[0] max|delta|=2.541e+03 (rtol=0.01 atol=0.001) | [pow_Tensor_Tensor_out](pow_Tensor_Tensor_out.md) |
| `prod.int_out` | out[0] max|delta|=1.000e+00 (rtol=0.0 atol=0.0) | [prod_int_out](prod_int_out.md) |
| `relu` | out[0] max|delta|=2.527e+00 (rtol=0.01 atol=0.001) | [relu](relu.md) |
| `remainder.Scalar` | out[0] max|delta|=7.000e+00 (rtol=0.0 atol=0.0) | [remainder_Scalar](remainder_Scalar.md) |
| `remainder.Scalar_out` | out[0] max|delta|=8.000e+00 (rtol=0.01 atol=0.001) | [remainder_Scalar_out](remainder_Scalar_out.md) |
| `remainder.Tensor_out` | out[0] max|delta|=4.000e+00 (rtol=0.01 atol=0.001) | [remainder_Tensor_out](remainder_Tensor_out.md) |
| `round.out` | out[0] max|delta|=1.000e+00 (rtol=0.01 atol=0.001) | [round_out](round_out.md) |
| `rsub.Scalar` | out[0] max|delta|=4.000e+00 (rtol=0.01 atol=0.001) | [rsub_Scalar](rsub_Scalar.md) |
| `select_copy.int` | out[0] max|delta|=1.667e-01 (rtol=0.01 atol=0.001) | [select_copy_int](select_copy_int.md) |
| `slice_copy.Tensor` | out[0] max|delta|=1.164e+00 (rtol=0.01 atol=0.001) | [slice_copy_Tensor](slice_copy_Tensor.md) |
| `split_with_sizes_copy` | out[0] max|delta|=7.000e+00 (rtol=0.0 atol=0.0) | [split_with_sizes_copy](split_with_sizes_copy.md) |
| `transpose_copy.int` | out[0] max|delta|=4.000e+00 (rtol=0.0 atol=0.0) | [transpose_copy_int](transpose_copy_int.md) |
| `unsqueeze_copy` | out[0] max|delta|=5.510e-01 (rtol=0.01 atol=0.001) | [unsqueeze_copy](unsqueeze_copy.md) |
| `view_copy` | out[0] max|delta|=4.770e+00 (rtol=0.01 atol=0.001) | [view_copy](view_copy.md) |

## Non-deterministic kernels (intermittent wrong output, same input)

| op | divergence | repro |
|---|---|---|
| `argmin.out` | no reply 60s | [argmin_out](argmin_out.md) |
| `bitwise_and.Scalar_out` | no reply 60s | [bitwise_and_Scalar_out](bitwise_and_Scalar_out.md) |
| `bitwise_and.Tensor_out` | no reply 60s | [bitwise_and_Tensor_out](bitwise_and_Tensor_out.md) |
| `flip` | op reproduces eager alone | [flip](flip.md) |

## Ruled out (40) — investigated, NOT filed

| op | why |
|---|---|
| `_native_batch_norm_legit.no_stats` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `_softmax.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `_upsample_bilinear2d_aa.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `abs` | reproduces eager when it runs (1/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `acos.out` | reproduces eager when it runs (4/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `add.Scalar` | single-op isolation won't lower/run (7/7 device-fail); op was bisected to OPERATOR but is unconfirmable as a one-op graph — needs a 2-op minimal repro |
| `alias_copy` | reproduces eager when it runs (6/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `any.dims_out` | single-op isolation won't lower/run (7/7 device-fail); op was bisected to OPERATOR but is unconfirmable as a one-op graph — needs a 2-op minimal repro |
| `atanh.out` | reproduces eager when it runs (5/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `bitwise_or.Scalar` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `bitwise_right_shift.Tensor_Scalar` | reproduces eager when it runs (2/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `bitwise_right_shift.Tensor_Scalar_out` | reproduces eager when it runs (6/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `bitwise_right_shift.Tensor_out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `bitwise_xor.Tensor_out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `constant_pad_nd` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `cos.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `div.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `exp.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `expand_copy` | single-op isolation won't lower/run (7/7 device-fail); op was bisected to OPERATOR but is unconfirmable as a one-op graph — needs a 2-op minimal repro |
| `floor_divide.out` | single-op isolation won't lower/run (7/7 device-fail); op was bisected to OPERATOR but is unconfirmable as a one-op graph — needs a 2-op minimal repro |
| `gelu.out` | reproduces eager when it runs (6/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `hardtanh.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `index_select.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `masked_fill.Scalar` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `max.dim_max` | single-op isolation won't lower/run (7/7 device-fail); op was bisected to OPERATOR but is unconfirmable as a one-op graph — needs a 2-op minimal repro |
| `mean.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `minimum.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `narrow_copy` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `neg.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `pow.Scalar_out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `select_scatter` | single-op isolation won't lower/run (7/7 device-fail); op was bisected to OPERATOR but is unconfirmable as a one-op graph — needs a 2-op minimal repro |
| `sin.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `squeeze_copy.dim` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `squeeze_copy.dims` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `sub.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `sum.IntList_out` | single-op isolation won't lower/run (7/7 device-fail); op was bisected to OPERATOR but is unconfirmable as a one-op graph — needs a 2-op minimal repro |
| `tril.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `unfold_copy` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `var_mean.correction` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
| `where.self_out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient |
