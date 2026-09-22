# Operator bugs — Vulkan on ASUS a12201

Device-verified by single-op isolation with EXACT baked inputs (one-op graph fed the op's real runtime values, lowered to Vulkan, run on the ASUS; diverges as the sole output with finite inputs → genuine kernel bug). 195 OPERATOR-verdict candidates → 42 isolated GENUINE → **24 confirmed** after 7× re-verification (18 ruled out). Each has a runnable `repro_<op>.py`.

- **21 confirmed operator bugs**
- **3 non-deterministic kernels** (intermittent wrong output, same input)
- **18 ruled out** (`ruled_out/`)

## Confirmed operator bugs

### wrong value (21)

| op | divergence | repro |
|---|---|---|
| `_upsample_bilinear2d_aa.out` | out[0] max|delta|=5.000e+00 (rtol=0.0 atol=0.0) | [_upsample_bilinear2d_aa_out](_upsample_bilinear2d_aa_out.md) |
| `bitwise_left_shift.Tensor_Scalar` | out[0] max|delta|=8.647e+17 (rtol=0.0 atol=0.0) | [bitwise_left_shift_Tensor_Scalar](bitwise_left_shift_Tensor_Scalar.md) |
| `bitwise_left_shift.Tensor_Scalar_out` | out[0] max|delta|=2.147e+09 (rtol=0.01 atol=0.001) | [bitwise_left_shift_Tensor_Scalar_out](bitwise_left_shift_Tensor_Scalar_out.md) |
| `bitwise_left_shift.Tensor_out` | out[0] max|delta|=9.223e+18 (rtol=0.0 atol=0.0) | [bitwise_left_shift_Tensor_out](bitwise_left_shift_Tensor_out.md) |
| `bitwise_right_shift.Tensor_Scalar` | out[0] max|delta|=5.120e+03 (rtol=0.0 atol=0.0) | [bitwise_right_shift_Tensor_Scalar](bitwise_right_shift_Tensor_Scalar.md) |
| `bitwise_right_shift.Tensor_Scalar_out` | out[0] max|delta|=1.600e+01 (rtol=0.01 atol=0.001) | [bitwise_right_shift_Tensor_Scalar_out](bitwise_right_shift_Tensor_Scalar_out.md) |
| `clamp.Tensor_out` | out[0] max|delta|=1.200e+01 (rtol=0.0 atol=0.0) | [clamp_Tensor_out](clamp_Tensor_out.md) |
| `copy` | out[0] max|delta|=2.550e+02 (rtol=0.0 atol=0.0) | [copy](copy.md) |
| `cos.out` | out[0] max|delta|=1.012e+00 (rtol=0.01 atol=0.001) | [cos_out](cos_out.md) |
| `div.out_mode` | out[0] dtype torch.float16 vs torch.float32 | [div_out_mode](div_out_mode.md) |
| `fill.Scalar` | out[0] max|delta|=1.000e+00 (rtol=0.0 atol=0.0) | [fill_Scalar](fill_Scalar.md) |
| `index_put` | out[0] max|delta|=1.000e+00 (rtol=0.0 atol=0.0) | [index_put](index_put.md) |
| `min.dim_min` | out[0] max|delta|=9.223e+18 (rtol=0.0 atol=0.0) | [min_dim_min](min_dim_min.md) |
| `min.unary_out` | out[0] max|delta|=1.246e+11 (rtol=0.01 atol=0.001) | [min_unary_out](min_unary_out.md) |
| `mul.out` | out[0] max|delta|=5.042e+21 (rtol=0.01 atol=0.001) | [mul_out](mul_out.md) |
| `pow.Tensor_Scalar_out` | out[0] max|delta|=3.286e+02 (rtol=0.01 atol=0.001) | [pow_Tensor_Scalar_out](pow_Tensor_Scalar_out.md) |
| `pow.Tensor_Tensor_out` | out[0] max|delta|=2.541e+03 (rtol=0.01 atol=0.001) | [pow_Tensor_Tensor_out](pow_Tensor_Tensor_out.md) |
| `prod.int_out` | out[0] dtype torch.int32 vs torch.int64 | [prod_int_out](prod_int_out.md) |
| `remainder.Scalar` | out[0] max|delta|=7.000e+00 (rtol=0.0 atol=0.0) | [remainder_Scalar](remainder_Scalar.md) |
| `remainder.Scalar_out` | out[0] max|delta|=8.000e+00 (rtol=0.01 atol=0.001) | [remainder_Scalar_out](remainder_Scalar_out.md) |
| `remainder.Tensor_out` | out[0] max|delta|=4.000e+00 (rtol=0.01 atol=0.001) | [remainder_Tensor_out](remainder_Tensor_out.md) |

## Non-deterministic kernels

| op | divergence | repro |
|---|---|---|
| `asin.out` | out[0] dtype torch.float32 vs torch.int64 | [asin_out](asin_out.md) |
| `bitwise_or.Tensor_out` | no reply 60s | [bitwise_or_Tensor_out](bitwise_or_Tensor_out.md) |
| `clamp.out` | out[0] max|delta|=2.000e+00 (rtol=0.01 atol=0.001) | [clamp_out](clamp_out.md) |

## Ruled out (18)

| op | why |
|---|---|
| `_native_batch_norm_legit.no_stats` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug |
| `abs` | reproduces eager when it runs (6/7 clean) — NOT a kernel bug |
| `acos.out` | reproduces eager when it runs (5/7 clean) — NOT a kernel bug |
| `any.all_out` | single-op isolation won't lower/run (7/7 device-fail); bisected OPERATOR but unconfirmable as one op |
| `any.dims_out` | single-op isolation won't lower/run (7/7 device-fail); bisected OPERATOR but unconfirmable as one op |
| `atanh.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug |
| `bitwise_right_shift.Tensor_out` | reproduces eager when it runs (5/7 clean) — NOT a kernel bug |
| `bitwise_xor.Tensor_out` | reproduces eager when it runs (5/7 clean) — NOT a kernel bug |
| `floor_divide` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug |
| `floor_divide.out` | single-op isolation won't lower/run (7/7 device-fail); bisected OPERATOR but unconfirmable as one op |
| `gather.out` | reproduces eager when it runs (6/7 clean) — NOT a kernel bug |
| `gt.Tensor_out` | single-op isolation won't lower/run (7/7 device-fail); bisected OPERATOR but unconfirmable as one op |
| `index_select.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug |
| `mean.out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug |
| `pow.Scalar_out` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug |
| `select_scatter` | single-op isolation won't lower/run (7/7 device-fail); bisected OPERATOR but unconfirmable as one op |
| `sum.IntList_out` | single-op isolation won't lower/run (7/7 device-fail); bisected OPERATOR but unconfirmable as one op |
| `var_mean.correction` | reproduces eager when it runs (7/7 clean) — NOT a kernel bug |
