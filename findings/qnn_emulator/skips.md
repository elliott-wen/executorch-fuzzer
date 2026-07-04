# SKIP + CRASH reasons (QNN HTP x86 emulator)

## CRASH — `qnn_executor_runner` aborts / segfaults

Unlike the phones (opaque "native abort"), the QNN emulator gives a real signal: the runner
process **Aborts** (444) or **Segfaults** (165) with a core dump. Top crashing ops:

| count | operator |
|---:|---|
| 329 | `stack` |
| 67 | `replication_pad3d.out` |
| 48 | `max_pool2d_with_indices_backward.grad_input` |
| 45 | `replication_pad2d.out` |
| 33 | `reflection_pad2d` |
| 32 | `copy` |
| 25 | `reflection_pad2d.out` |
| 25 | `split_with_sizes_copy` |
| 3 | `narrow_copy` |
| 2 | `unfold_copy` |

## SKIP — `qnn_runner rc=2` (graph not runnable on the HTP target)

All QNN skips return exit code 2 (unsupported/partition-rejected on HTP) with no further detail —
so the finding is the **op list** that the HTP cannot run. Top:

| count | operator |
|---:|---|
| 593 | `native_group_norm` |
| 400 | `pixel_shuffle` |
| 371 | `narrow_copy` |
| 336 | `any.dims_out` |
| 331 | `_fft_r2c` |
| 321 | `_pdist_forward` |
| 287 | `pixel_unshuffle` |
| 223 | `copy` |
| 188 | `topk.values` |
| 177 | `var.correction_out` |
| 156 | `narrow_copy.out` |
| 122 | `mean.dtype_out` |
| 117 | `any.all_out` |
| 117 | `scatter_add.out` |
| 110 | `min.unary_out` |
| 106 | `max.unary_out` |
