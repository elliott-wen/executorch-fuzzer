# OpenVINO single-op — SKIP & CRASH reason tables

Every SKIP/CRASH is one operator (single-op corpus). The *reason* is the finding.


## SKIP class A — OpenVINO delegate blob failure (coverage gap)

Runtime reason (verbatim, all identical modulo the hex code):

```
client RuntimeError: method->execute() failed with error 0x1 | [OpenvinoBackend.cpp:98] OpenVINO: runtime loaded successfully ... | [method.cpp:1525] CALL_DELEGATE execute failed at instruction 0: 0x1
```

The op partitions to OpenVINO and its blob compiles, but the compiled model **fails at execute** — a real OpenVINO coverage gap. Ops (count desc):

| count | operator |
|--:|---|
| 437 | `prod` |
| 434 | `zeros.out` |
| 433 | `full.out` |
| 433 | `any.all_out` |
| 433 | `mean` |
| 431 | `ones.out` |
| 431 | `index_put` |
| 430 | `mean.dtype_out` |
| 421 | `prod.out` |
| 394 | `linear` |
| 391 | `max` |
| 388 | `argmin.out` |
| 385 | `masked_fill.Scalar` |
| 384 | `min.unary_out` |
| 383 | `argmax.out` |
| 380 | `min` |
| 380 | `pixel_unshuffle` |
| 374 | `max.unary_out` |
| 337 | `select_copy.int` |
| 252 | `var.correction_out` |
| 246 | `squeeze_copy.dims` |
| 230 | `var_mean.correction` |
| 216 | `var.correction` |
| 188 | `sum.IntList_out` |
| 172 | `amax.out` |
| 148 | `amin.out` |
| 141 | `arange.out` |
| 137 | `div.Scalar` |
| 95 | `erf.out` |
| 89 | `linear.out` |
| 71 | `prod.int_out` |
| 29 | `stack` |
| 26 | `scatter.value_out` |
| 26 | `view_copy` |
| 24 | `topk.values` |
| 22 | `any.dims_out` |
| 17 | `cumsum.out` |
| 12 | `min.dim_min` |
| 9 | `where.self_out` |
| 9 | `max.dim_max` |
| 9 | `index_select.out` |
| 9 | `bitwise_and.Scalar_out` |
| 7 | `fmod.Tensor_out` |
| 7 | `gt.Tensor_out` |
| 7 | `mul.Scalar` |
| 7 | `alias_copy` |
| 7 | `arange.start_out` |
| 7 | `repeat` |
| 6 | `unsqueeze_copy` |
| 6 | `t_copy` |
| 6 | `pow.Tensor_Tensor_out` |
| 6 | `minimum.out` |
| 6 | `eq.Scalar_out` |
| 6 | `add.Scalar` |
| 6 | `hardtanh` |
| 6 | `_softmax.out` |
| 5 | `pow.Tensor_Scalar_out` |
| 5 | `lt.Tensor_out` |
| 5 | `ge.Tensor_out` |
| 5 | `_adaptive_avg_pool2d` |
| 5 | `maximum.out` |
| 5 | `eq.Tensor_out` |
| 5 | `isinf` |
| 5 | `fill.Scalar` |
| 5 | `glu.out` |
| 5 | `squeeze_copy.dim` |
| 4 | `logical_not` |
| 4 | `div.out` |
| 4 | `fmod.Scalar` |
| 4 | `asinh.out` |
| 4 | `pow.Scalar_out` |
| 4 | `addmm.out` |
| 4 | `lift_fresh_copy` |
| 4 | `copy` |
| 4 | `relu` |
| 4 | `div.out_mode` |
| 4 | `roll` |
| 3 | `asin.out` |
| 3 | `select_scatter` |
| 3 | `abs` |
| 3 | `le.Tensor_out` |
| 3 | `flip` |
| 3 | `clamp.Tensor_out` |
| 3 | `ne.Scalar_out` |
| 3 | `bitwise_left_shift.Tensor_Scalar_out` |
| 3 | `sigmoid.out` |
| 3 | `leaky_relu.out` |
| 3 | `floor_divide.out` |
| 3 | `fmod.Scalar_out` |
| 2 | `reciprocal.out` |
| 2 | `sinh.out` |
| 2 | `rsub.Scalar` |
| 2 | `bitwise_right_shift.Tensor_out` |
| 2 | `acos.out` |
| 2 | `bitwise_or.Scalar_out` |
| 2 | `bitwise_not.out` |
| 2 | `gt.Scalar_out` |
| 2 | `floor_divide` |
| 2 | `bitwise_and.Tensor_out` |
| 2 | `ge.Scalar_out` |
| 2 | `mul.out` |
| 2 | `bitwise_xor.Tensor_out` |
| 2 | `cosh.out` |
| 2 | `constant_pad_nd` |
| 2 | `bitwise_and.Scalar` |
| 2 | `log1p.out` |
| 2 | `logit` |
| 2 | `bitwise_left_shift.Tensor_out` |
| 2 | `logit.out` |
| 2 | `cos.out` |
| 2 | `exp.out` |
| 1 | `sin.out` |
| 1 | `abs.out` |
| 1 | `sub.out` |
| 1 | `clamp.out` |
| 1 | `tril.out` |
| 1 | `diagonal_copy` |
| 1 | `isnan` |
| 1 | `sqrt.out` |
| 1 | `bitwise_right_shift.Tensor_Scalar_out` |
| 1 | `rsqrt.out` |
| 1 | `bitwise_xor.Scalar_out` |
| 1 | `log2.out` |
| 1 | `native_layer_norm` |
| 1 | `permute_copy` |
| 1 | `ne.Tensor_out` |
| 1 | `_log_softmax.out` |
| 1 | `tan.out` |
| 1 | `floor.out` |
| 1 | `logical_xor.out` |
| 1 | `gelu.out` |
| 1 | `hardtanh.out` |
| 1 | `transpose_copy.int` |
| 1 | `log10.out` |

**Total OV-delegate SKIPs: 10143 across 134 operators.**


## SKIP class B — portable-kernel rejection (error 0x12, NOT OpenVINO)

These ops fell back to portable CPU kernels which then raised a `Check failed` / unsupported-dtype / unsupported-arg guard. Per step 3a these are **portable-side coverage gaps, not OpenVINO bugs** — listed with the verbatim guard clause.

| count | operator | portable guard (verbatim) |
|--:|---|---|
| 296 | `_pdist_forward` | portable tensor_util.h: Check failed (static_cast<size_t>(t.dim() |
| 237 | `_fft_r2c` | portable tensor_util_portable.cpp: Check failed (all_contiguous || all_channels_last) |
| 130 | `_fft_r2c` | portable op_fft_r2c.cpp: Check failed (onesided) |
| 130 | `scatter_add.out` | portable index_util.cpp: Check failed (index.scalar_type() |
| 20 | `_pdist_forward` | portable op_pdist_forward.cpp: Unhandled dtype Long |
| 19 | `sub.Scalar` | portable op_sub.cpp: Check failed (utils::extract_scalar(alpha, &alpha_val) |
| 17 | `_pdist_forward` | portable op_pdist_forward.cpp: Unhandled dtype Int |
| 16 | `_pdist_forward` | portable op_pdist_forward.cpp: Unhandled dtype Bool |
| 1 | `narrow_copy` | portable slice_util.cpp: Check failed (start >= 0 && length >= 0 && step >= 0) |

**Total portable-rejection SKIPs: 866 across 5 operators.**


## CRASH — native aborts

All CRASHes logged `executor died (native abort)` — a native SIGABRT/SIGSEGV with **no catchable error message**. The missing guard is the bug (the kernel should reject the form with a catchable error, not abort).


### CRASH on the OpenVINO delegate (ops>=1) — real finding

| count | operator | trigger |
|--:|---|---|
| 55 | `max_pool2d_with_indices_backward.grad_input` | native abort (no catchable message) |
| 1 | `unfold_copy` | native abort (no catchable message) |
| 1 | `reflection_pad2d.out` | native abort (no catchable message) |
| 1 | `convolution` | native abort (no catchable message) |

**Total delegated CRASHes: 58 across 4 operators.**


### CRASH on portable fallback (ops=0) — ruled out (not OpenVINO)

| count | operator |
|--:|---|
| 9 | `narrow_copy` |
| 2 | `narrow_copy.out` |

**Total portable CRASHes: 11 across 2 operators.**
