# Core ML (Apple-Silicon Mac) — SKIP and CRASH reason tables

The reason **is** the finding for these modes. SKIP = the op lowered into the corpus
(passed partitioning) but the Core ML runtime **rejected it at execution**; a real coverage
gap. CRASH = `executor died (native abort)` with no catchable message; the crashing form is
recorded by corpus job id. All counts are from the full 52,970-graph run; CRASH rows are the
**re-run-confirmed** ones (collateral removed — see bugs/crash-native-abort.md).

## SKIP — 3653 rows across 65 operators

Aggregate reason signatures:

| count | runtime reason (verbatim clause) |
|------:|-----------------------------------|
| 1377 | `Failed to compile model` |
| 1075 | `Caught an unknown exception` |
| 316 | `Attempted to resize a static tensor` |
| 305 | `Check failed (static_cast<size_t>(t.dim(...)` |
| 245 | `different dim orders` |
| 126 | `Model init failed` |
| 123 | `onesided=False is not supported yet` |
| 25 | `DataType=1 is not supported` |
| 17 | `Unhandled dtype Long` |
| 16 | `Unhandled dtype Bool` |
| 14 | `Unhandled dtype Int` |
| 13 | `Failed to prewarm model` |
| 1 | `Check failed (start >= 0 && length >= 0 && step >= 0...)` |

### Per-operator SKIP (top reason shown; some ops SKIP for several reasons)

| count | operator | primary runtime reason |
|------:|----------|------------------------|
| 549 | `any.dims_out` | `Caught an unknown exception` (×364) (+1 other reason) |
| 429 | `index_put` | `Failed to compile model` (×416) (+1 other reason) |
| 368 | `_fft_r2c` | `different dim orders` (×245) (+1 other reason) |
| 352 | `_pdist_forward` | `Check failed (static_cast<size_t>(t.dim(...)` (×305) (+3 other reason) |
| 276 | `var.correction_out` | `Caught an unknown exception` (×222) (+1 other reason) |
| 236 | `mean.out` | `Caught an unknown exception` (×127) (+1 other reason) |
| 217 | `avg_pool2d.out` | `Attempted to resize a static tensor` (×217) |
| 135 | `_softmax.out` | `Failed to compile model` (×131) (+1 other reason) |
| 129 | `_log_softmax.out` | `Failed to compile model` (×125) (+1 other reason) |
| 118 | `split_with_sizes_copy` | `Model init failed` (×64) (+1 other reason) |
| 114 | `sum.IntList_out` | `Failed to compile model` (×108) (+2 other reason) |
| 106 | `any.all_out` | `Caught an unknown exception` (×96) (+1 other reason) |
| 81 | `mean.dtype_out` | `Caught an unknown exception` (×71) (+1 other reason) |
| 65 | `clamp.Tensor_out` | `Failed to compile model` (×64) (+1 other reason) |
| 63 | `min.unary_out` | `Caught an unknown exception` (×46) (+1 other reason) |
| 58 | `min.dim_min` | `Attempted to resize a static tensor` (×48) (+1 other reason) |
| 58 | `max.unary_out` | `Caught an unknown exception` (×38) (+1 other reason) |
| 46 | `max.dim_max` | `Attempted to resize a static tensor` (×41) (+1 other reason) |
| 31 | `pixel_shuffle` | `Model init failed` (×31) |
| 24 | `prod.int_out` | `Caught an unknown exception` (×17) (+1 other reason) |
| 22 | `index_select.out` | `Caught an unknown exception` (×14) (+1 other reason) |
| 22 | `split_copy.Tensor` | `Failed to compile model` (×22) |
| 18 | `index_select` | `Caught an unknown exception` (×16) (+1 other reason) |
| 16 | `amin.out` | `Failed to compile model` (×16) |
| 15 | `amax.out` | `Failed to compile model` (×15) |
| 14 | `convolution` | `Failed to compile model` (×14) |
| 11 | `mean` | `Model init failed` (×8) (+1 other reason) |
| 6 | `replication_pad2d.out` | `Failed to compile model` (×6) |
| 6 | `linear.out` | `Caught an unknown exception` (×6) |
| 5 | `max` | `Model init failed` (×5) |
| 5 | `var.correction` | `Caught an unknown exception` (×5) |
| 4 | `mul.Scalar` | `Caught an unknown exception` (×4) |
| 4 | `bitwise_left_shift.Tensor_out` | `Caught an unknown exception` (×2) (+1 other reason) |
| 4 | `bitwise_xor.Tensor_out` | `DataType=1 is not supported` (×4) |
| 4 | `bitwise_right_shift.Tensor_out` | `DataType=1 is not supported` (×3) (+1 other reason) |
| 3 | `unsqueeze_copy` | `Caught an unknown exception` (×3) |
| 3 | `bitwise_or.Tensor_out` | `DataType=1 is not supported` (×3) |
| 2 | `masked_fill.Scalar` | `Caught an unknown exception` (×2) |
| 2 | `embedding` | `Caught an unknown exception` (×2) |
| 2 | `abs` | `Caught an unknown exception` (×2) |
| 2 | `div.out_mode` | `DataType=1 is not supported` (×2) |
| 2 | `cumsum.out` | `Caught an unknown exception` (×2) |
| 2 | `flip` | `Model init failed` (×2) |
| 2 | `div.Scalar` | `Caught an unknown exception` (×2) |
| 2 | `logit` | `Caught an unknown exception` (×2) |
| 1 | `remainder.Scalar` | `Caught an unknown exception` (×1) |
| 1 | `glu.out` | `Caught an unknown exception` (×1) |
| 1 | `erf.out` | `Caught an unknown exception` (×1) |
| 1 | `bmm.out` | `Caught an unknown exception` (×1) |
| 1 | `logical_not` | `Caught an unknown exception` (×1) |
| 1 | `round.out` | `Caught an unknown exception` (×1) |
| 1 | `isnan` | `Caught an unknown exception` (×1) |
| 1 | `bitwise_right_shift.Tensor_Scalar_out` | `Caught an unknown exception` (×1) |
| 1 | `topk.values` | `Failed to compile model` (×1) |
| 1 | `log2.out` | `Caught an unknown exception` (×1) |
| 1 | `narrow_copy` | `Check failed (start >= 0 && length >= 0 && step >= 0...)` (×1) |
| 1 | `relu` | `Caught an unknown exception` (×1) |
| 1 | `le.Tensor_out` | `DataType=1 is not supported` (×1) |
| 1 | `ne.Scalar_out` | `Caught an unknown exception` (×1) |
| 1 | `where.self_out` | `Model init failed` (×1) |
| 1 | `neg.out` | `Caught an unknown exception` (×1) |
| 1 | `eq.Tensor_out` | `Caught an unknown exception` (×1) |
| 1 | `ge.Tensor_out` | `Caught an unknown exception` (×1) |
| 1 | `bitwise_left_shift.Tensor_Scalar_out` | `Caught an unknown exception` (×1) |
| 1 | `transpose_copy.int` | `Caught an unknown exception` (×1) |

## CRASH — confirmed native aborts / hangs (crashing form by op)

Native aborts have **no catchable error string** — only `executor died (native abort)`.
Reproduction = distinct crashing samples that aborted again on the clean window=1 re-run
(or N/5 same-job for the gated ones).

| operator | reproduced | mode |
|----------|-----------:|------|
| `unfold_copy` | 25/25 | native abort |
| `prod.int_out` | 17/17 | native abort |
| `diagonal_copy` | 17/17 | native abort |
| `scatter.value_out` | 16/16 | native abort |
| `max_pool2d_with_indices_backward.grad_input` | 14/14 | native abort |
| `linear.out` | 5/5 | native abort |
| `mean.out` | 3/3 | native abort |
| `cumsum.out` | 5/5 gate | native abort |
| `logical_or.out` | 5/5 gate | native abort |
| `mm.out` | 5/5 gate | native abort |
| `t_copy` | 5/5 gate | native abort |
| `tril.out` | 5/5 gate | native abort |
| `alias_copy` | 5/5 gate | native abort |
| `view_copy` | 4/5 gate | native abort |
| `slice_scatter` | 16/17 | HANG (timeout) |
| `select_scatter` | 9/9 | HANG (timeout) |
| `max.unary_out` | 5/5 gate | HANG (timeout) |
