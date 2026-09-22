# SKIP reasons per operator (portable, 87,921-graph run)

SKIP = the portable `.pte` loaded but the device **rejected it at runtime**. These are coverage gaps
of the portable backend (a guard/kernel that refuses the form), NOT correctness bugs. Each row is the
runtime `Check failed`/error the portable kernel raised. Enumerated per the `analysis_single.md` Skip
step.

Notable recurring classes:
- **int64-index requirement** (`scatter*`, `gather` — 232): portable index kernels reject `Int`
  indices, require `int64` (`Expected dtype int64 for index; index.scalar_type() = Int`).
- **scalar-extraction guards** (`arange`, `sub.Scalar`, `rsub.Scalar`): `extract_scalar(...)` fails
  on the scalar arg form.
- **rank/shape guards** (`_pdist_forward` rank!=2, `convolution` 5-D input, `_adaptive_avg_pool2d`
  output_size 0, `narrow_copy` negative args).
- **unimplemented forms** (`copy` non_blocking=true, `expand_copy` implicit=true).
- **dim-order** (`_fft_r2c` mixed dim orders).

| count | operator | runtime reason (representative) |
|---:|---|---|
| 738 | `native_group_norm` | [normalization_ops_util.cpp:160] Check failed (in.size(0) == N): _(+1 other form(s))_ |
| 427 | `arange.start_out` | [op_arange.cpp:55] Check failed (utils::extract_scalar(start, &d_start)): _(+2 other form(s))_ |
| 350 | `_fft_r2c` | [tensor_util_portable.cpp:130] Check failed (all_contiguous \|\| all_channels_last): 2 input tensors have different dim orders _(+1 other form(s))_ |
| 347 | `_pdist_forward` | [distance_util.cpp:16] Check failed (tensor_is_rank(in, 2)): _(+1 other form(s))_ |
| 219 | `copy` | [op_copy.cpp:33] Check failed (non_blocking == false): |
| 207 | `expand_copy` | [copy_ops_util.cpp:165] Check failed (implicit == false): This operator is not implemented for when implicit == true. |
| 191 | `_log_softmax.out` | [op_log_softmax.cpp:152] Check failed (false): |
| 115 | `scatter_add.out` | [index_util.cpp:156] Check failed (index.scalar_type() == ScalarType::Long): Expected dtype int64 for index; index.scalar_type() = Int |
| 88 | `scatter.src_out` | [index_util.cpp:156] Check failed (index.scalar_type() == ScalarType::Long): Expected dtype int64 for index; index.scalar_type() = Int |
| 56 | `linear.out` | client RuntimeError: method->execute() failed with error 0x12 \| [op_permute_copy |
| 50 | `add.Scalar` | [op_add.cpp:118] Check failed ((common_type == a_type && check_alpha_type(utils::get_scalar_dtype(alpha), common_type))): |
| 31 | `sub.Scalar` | [op_sub.cpp:144] Check failed (utils::extract_scalar(alpha, &alpha_val)): |
| 24 | `scatter.value_out` | [index_util.cpp:24] Check failed (index.scalar_type() == ScalarType::Long): Expected dtype int64 for index; index.scalar_type() = Int |
| 22 | `convolution` | [kernel_ops_util.cpp:389] Check failed (in.dim() == 3 \|\| in.dim() == 4): Expect input tensor to be 3-D or 4-D, but got, 5. |
| 14 | `stack` | client RuntimeError: method->execute() failed with error 0x12 \| [op_cat.cpp:88] _(+1 other form(s))_ |
| 5 | `gather.out` | [index_util.cpp:24] Check failed (index.scalar_type() == ScalarType::Long): Expected dtype int64 for index; index.scalar_type() = Int |
| 4 | `rsub.Scalar` | [op_sub.cpp:78] Check failed (utils::extract_scalar(alpha, &alpha_val)): |
| 3 | `_adaptive_avg_pool2d` | [kernel_ops_util.cpp:285] Check failed (output_size[0] > 0 && output_size[1] > 0): output_size must be positive, but got (0, 0) |
| 2 | `_native_batch_norm_legit_no_training` | [normalization_ops_util.cpp:39] Check failed (tensors_have_same_dtype(in, running_mean.value())): |
| 2 | `narrow_copy` | [slice_util.cpp:173] Check failed (start >= 0 && length >= 0 && step >= 0): Input args should be >= 0. |
| 1 | `pixel_unshuffle` | client RuntimeError: method->execute() failed with error 0x12 \| [op_permute_copy |
| 1 | `mul.out` | [binary_ops.cpp:36] Check failed (error == Error::Ok): Failed to resize output tensor. |

---

# CRASH reasons per operator

CRASH = native abort (the executor process died) — there is **no catchable error string**; the runtime
returns only `executor died (native abort)`. The "reason" is therefore the crashing **operator + input
form** (the kernel aborts instead of validating). All are from the 87,921-graph run.

| count | operator | reason / trigger |
|---:|---|---|
| 51 | `max_pool2d_with_indices_backward.grad_input` | native abort; triggered by a non-finite input element (no bounds/finite guard in the backward scatter) |
| 4 | `unfold_copy` | native abort; degenerate shape / non-finite input (no guard) |
| 4 | `narrow_copy` | native abort; non-finite / out-of-range args reach the copy kernel |
| 1 | `narrow_copy.out` | native abort (same as `narrow_copy`) |
| 1 | `_log_softmax.out` | native abort (rare; distinct from its 191 runtime-SKIPs) |

Note: a native abort should ideally be a catchable `Check failed` SKIP instead — the crash *is* the
bug (the kernel lacks the guard the SKIP-path kernels have).
