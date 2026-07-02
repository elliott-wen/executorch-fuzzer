# Operator bugs — QNN HTP x86 emulator (fp16)

Device-verified by single-op isolation with EXACT baked inputs on the QNN HTP x86 emulator (qualcomm fp16 backend). 194 OPERATOR-verdict candidates → 58 isolated GENUINE → **32 confirmed** after 7× re-verification (26 ruled out). Each has a `repro_<op>.py`.

- **25 confirmed operator bugs**
- **7 non-deterministic kernels**
- **26 ruled out** (`ruled_out/`)

## Confirmed operator bugs

### wrong value (25)

| op | divergence | repro |
|---|---|---|
| `_log_softmax.out` | out[0] shape (1,) vs (3, 2, 2, 4, 4) | [_log_softmax_out](_log_softmax_out.md) |
| `_native_batch_norm_legit.no_stats` | out[0] dtype torch.float32 vs torch.uint8 | [_native_batch_norm_legit_no_stats](_native_batch_norm_legit_no_stats.md) |
| `_upsample_bilinear2d_aa.out` | out[0] max|delta|=2.000e+00 (rtol=0.0 atol=0.0) | [_upsample_bilinear2d_aa_out](_upsample_bilinear2d_aa_out.md) |
| `bitwise_left_shift.Tensor_Scalar` | out[0] max|delta|=1.441e+17 (rtol=0.0 atol=0.0) | [bitwise_left_shift_Tensor_Scalar](bitwise_left_shift_Tensor_Scalar.md) |
| `bitwise_left_shift.Tensor_Scalar_out` | out[0] dtype torch.float32 vs torch.uint8 | [bitwise_left_shift_Tensor_Scalar_out](bitwise_left_shift_Tensor_Scalar_out.md) |
| `bitwise_left_shift.Tensor_out` | out[0] dtype torch.int64 vs torch.uint8 | [bitwise_left_shift_Tensor_out](bitwise_left_shift_Tensor_out.md) |
| `bitwise_or.Scalar_out` | out[0] shape () vs (2, 3) | [bitwise_or_Scalar_out](bitwise_or_Scalar_out.md) |
| `bitwise_right_shift.Tensor_Scalar_out` | out[0] max|delta|=3.000e+00 (rtol=0.01 atol=0.001) | [bitwise_right_shift_Tensor_Scalar_out](bitwise_right_shift_Tensor_Scalar_out.md) |
| `bitwise_xor.Scalar_out` | no reply 60s | [bitwise_xor_Scalar_out](bitwise_xor_Scalar_out.md) |
| `elu.out` | out[0] max|delta|=1.464e+01 (rtol=0.01 atol=0.001) | [elu_out](elu_out.md) |
| `floor_divide` | out[0] max|delta|=1.000e+00 (rtol=0.01 atol=0.001) | [floor_divide](floor_divide.md) |
| `fmod.Scalar` | out[0] max|delta|=2.520e+02 (rtol=0.0 atol=0.0) | [fmod_Scalar](fmod_Scalar.md) |
| `fmod.Scalar_out` | no reply 60s | [fmod_Scalar_out](fmod_Scalar_out.md) |
| `index_select.out` | out[0] shape (2, 2, 4) vs (2, 1, 4, 2) | [index_select_out](index_select_out.md) |
| `minimum.out` | out[0] max|delta|=3.000e+00 (rtol=0.01 atol=0.001) | [minimum_out](minimum_out.md) |
| `neg.out` | op reproduces eager alone | [neg_out](neg_out.md) |
| `pow.Scalar_out` | out[0] max|delta|=3.770e+09 (rtol=0.01 atol=0.001) | [pow_Scalar_out](pow_Scalar_out.md) |
| `pow.Tensor_Tensor_out` | out[0] max|delta|=1.000e+00 (rtol=0.01 atol=0.001) | [pow_Tensor_Tensor_out](pow_Tensor_Tensor_out.md) |
| `prod.int_out` | out[0] max|delta|=1.000e+00 (rtol=0.0 atol=0.0) | [prod_int_out](prod_int_out.md) |
| `remainder.Scalar` | out[0] max|delta|=8.000e+00 (rtol=0.0 atol=0.0) | [remainder_Scalar](remainder_Scalar.md) |
| `replication_pad2d.out` | out[0] max|delta|=7.310e-01 (rtol=0.01 atol=0.001) | [replication_pad2d_out](replication_pad2d_out.md) |
| `roll` | out[0] max|delta|=1.000e+00 (rtol=0.0 atol=0.0) | [roll](roll.md) |
| `rsub.Scalar` | out[0] max|delta|=1.000e+00 (rtol=0.01 atol=0.001) | [rsub_Scalar](rsub_Scalar.md) |
| `sub.out` | out[0] max|delta|=4.109e+00 (rtol=0.01 atol=0.001) | [sub_out](sub_out.md) |
| `topk.values` | out[0] max|delta|=1.417e+00 (rtol=0.01 atol=0.001) | [topk_values](topk_values.md) |

## Non-deterministic kernels

| op | divergence | repro |
|---|---|---|
| `alias_copy` | op reproduces eager alone | [alias_copy](alias_copy.md) |
| `amin.out` | out[0] shape () vs (2, 3) | [amin_out](amin_out.md) |
| `bitwise_xor.Tensor_out` | op reproduces eager alone | [bitwise_xor_Tensor_out](bitwise_xor_Tensor_out.md) |
| `div.Scalar` | op reproduces eager alone | [div_Scalar](div_Scalar.md) |
| `expm1.out` | op reproduces eager alone | [expm1_out](expm1_out.md) |
| `flip` | op reproduces eager alone | [flip](flip.md) |
| `gelu.out` | out[0] max|delta|=3.175e+04 (rtol=0.01 atol=0.001) | [gelu_out](gelu_out.md) |

## Ruled out (26)

`_softmax.out`, `addmm.out`, `amax.out`, `any.all_out`, `any.dims_out`, `asin.out`, `bitwise_and.Scalar_out`, `bitwise_right_shift.Tensor_out`, `copy`, `cos.out`, `div.out`, `div.out_mode`, `exp.out`, `floor.out`, `index_select`, `linear`, `logical_not`, `logical_not.out`, `max.unary_out`, `maximum.out`, `pow.Tensor_Scalar_out`, `scatter.src_out`, `select_scatter`, `sin.out`, `split_with_sizes_copy`, `sum.IntList_out`
