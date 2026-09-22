# QNN phone SKIP + CRASH

## CRASH (native abort) — top ops

| count | op |
|---:|---|
| 76 | `split_with_sizes_copy` |
| 50 | `max_pool2d_with_indices_backward.grad_input` |
| 6 | `copy` |
| 4 | `unfold_copy` |
| 4 | `narrow_copy` |
| 3 | `replication_pad2d.out` |
| 3 | `replication_pad3d.out` |
| 3 | `reflection_pad2d` |
| 2 | `reflection_pad2d.out` |
| 1 | `bitwise_left_shift.Tensor_Scalar_out` |

## SKIP (HTP-unsupported) — top ops

| count | op |
|---:|---|
| 647 | `native_group_norm` |
| 368 | `_fft_r2c` |
| 301 | `_pdist_forward` |
| 253 | `copy` |
| 121 | `scatter_add.out` |
| 85 | `reflection_pad2d.out` |
| 72 | `reflection_pad2d` |
| 64 | `linear.out` |
| 58 | `repeat` |
| 46 | `mean.out` |
| 38 | `expand_copy` |
| 30 | `scatter.value_out` |
| 25 | `convolution` |
| 21 | `scatter.src_out` |
