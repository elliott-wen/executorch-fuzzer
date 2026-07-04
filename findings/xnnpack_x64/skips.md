# SKIP + CRASH reasons per operator (xnnpack, 87,921-graph injected run)

SKIP = the `.pte` loaded but the device rejected it at runtime. For xnnpack these split into two
kinds: **XNNPACK-delegate** rejections (`xnn_status_*` / `XNNCompiler` / `XNNExecutor` — a real
partition-vs-runtime gap in the backend under test; see the `xnn-err` column) and
**portable-fallback** rejections (undelegated ops hitting the same portable kernel guards). The reason
string is the finding.

| count | xnn-err | operator | runtime reason (representative) |
|---:|---:|---|---|
| 739 | 0 | `native_group_norm` | [normalization_ops_util.cpp:160] Check failed (in.size(0) == N):  |
| 424 | 0 | `arange.start_out` | [op_arange.cpp:55] Check failed (utils::extract_scalar(start, &d_start)):  |
| 353 | 0 | `_fft_r2c` | [tensor_util_portable.cpp:130] Check failed (all_contiguous \|\| all_channels_last): 2 input tensors have different dim orders  |
| 347 | 0 | `_pdist_forward` | [distance_util.cpp:16] Check failed (tensor_is_rank(in, 2)):  |
| 219 | 0 | `copy` | [op_copy.cpp:33] Check failed (non_blocking == false):  |
| 207 | 0 | `expand_copy` | [copy_ops_util.cpp:165] Check failed (implicit == false): This operator is not implemented for when implicit == true.  |
| 199 | 0 | `_log_softmax.out` | [op_log_softmax.cpp:152] Check failed (false):  |
| 110 | 0 | `scatter_add.out` | [index_util.cpp:156] Check failed (index.scalar_type() == ScalarType::Long): Expected dtype int64 for index; index.scalar_type() = |
| 84 | 0 | `scatter.src_out` | [index_util.cpp:156] Check failed (index.scalar_type() == ScalarType::Long): Expected dtype int64 for index; index.scalar_type() = |
| 61 | 58 | `pixel_shuffle` | xnn_status_unsupported_parameter |
| 59 | 0 | `linear.out` | client RuntimeError: method->execute() failed with error 0x12 | [op_pe |
| 51 | 0 | `add.Scalar` | [op_add.cpp:118] Check failed ((common_type == a_type && check_alpha_type(utils::get_scalar_dtype(alpha), common_type))):  |
| 42 | 41 | `pixel_unshuffle` | xnn_status_unsupported_parameter |
| 33 | 0 | `scatter.value_out` | [index_util.cpp:24] Check failed (index.scalar_type() == ScalarType::Long): Expected dtype int64 for index; index.scalar_type() =  |
| 33 | 33 | `_softmax.out` | xnn_status_invalid_parameter |
| 31 | 0 | `sub.Scalar` | [op_sub.cpp:144] Check failed (utils::extract_scalar(alpha, &alpha_val)):  |
| 29 | 0 | `convolution` | [kernel_ops_util.cpp:389] Check failed (in.dim() == 3 \|\| in.dim() == 4): Expect input tensor to be 3-D or 4-D, but got, 5.  |
| 10 | 0 | `stack` | client RuntimeError: method->execute() failed with error 0x12 | [op_ca |
| 6 | 0 | `gather.out` | [index_util.cpp:24] Check failed (index.scalar_type() == ScalarType::Long): Expected dtype int64 for index; index.scalar_type() =  |
| 3 | 0 | `_adaptive_avg_pool2d` | [kernel_ops_util.cpp:285] Check failed (output_size[0] > 0 && output_size[1] > 0): output_size must be positive, but got (0, 0)  |
| 3 | 0 | `rsub.Scalar` | [op_sub.cpp:78] Check failed (utils::extract_scalar(alpha, &alpha_val)):  |
| 2 | 0 | `_native_batch_norm_legit_no_training` | [normalization_ops_util.cpp:39] Check failed (tensors_have_same_dtype(in, running_mean.value())):  |
| 1 | 0 | `var_mean.correction` | client RuntimeError: method->execute() failed with error 0x12 | [op_va |
| 1 | 0 | `cumsum.out` | [kernel_ops_util.cpp:512] Check failed (dim_is_valid(dim, in.dim())):  |

## CRASH (native abort — no catchable message; crashing op + trigger)

| count | operator |
|---:|---|
| 52 | `max_pool2d_with_indices_backward.grad_input` |
| 5 | `unfold_copy` |
| 3 | `narrow_copy` |
| 1 | `narrow_copy.out` |
