# Confirmed CUDA-backend findings (device-verified)

All single-operator, whole-graph-delegated (`ops≥1`), determinism-gated N=5 (5/5 reproduce).
Run any repro in `.venv-cuda` (`CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH
LD_LIBRARY_PATH=$CUDA_HOME/lib64 PYTHONPATH=/data/jwen929 .venv-cuda/bin/python bugs/repro.py <job_id>`).

| # | finding | mode | mechanism | ops | count | repro job |
|---|---|---|---|---|---:|---|
| 1 | [bitwise_shift](bitwise_shift.md) | MISMATCH | WRONG-VALUE: shift ≥ bitwidth / negative is UB in kernel | `bitwise_left/right_shift.Tensor_Scalar(.out)` | ~570 | w0:196, w0:237 |
| 2 | [int_reduce_overflow](int_reduce_overflow.md) | MISMATCH | WRONG-VALUE: integer prod/sum overflow differs | `prod.int_out`, `sum.IntList_out` | 37 | w0:226, w102:69 |
| 3 | [fp16_nonfinite](fp16_nonfinite.md) | MISMATCH | NONFINITE: fp16 overflow, CPU-NaN vs device-Inf | `gelu.out`, `floor_divide(.out)`, `grid_sampler_2d`, `elu.out` | ~134 | w0:117, w11:273 |
| 4 | [crash_maxpool2d_backward](crash_maxpool2d_backward.md) | CRASH | device-side assert index-OOB (missing input guard) | `max_pool2d_with_indices_backward.grad_input` | 281 | w0:625 |
| 5 | [crash_clamp_tensor](crash_clamp_tensor.md) | CRASH | illegal memory access (GPU OOB) | `clamp.Tensor_out` | 177 | w0:168 |
| 6 | [skip_slimtensor_shim](skip_slimtensor_shim.md) | SKIP | runtime rejects offset≠0 / non-contiguous / bool inputs | `select_copy.int`, `slice_copy.Tensor`, `any.*` | 794 | w0:33, w0:11, w0:400 |

Filtered confounds (not device bugs): [../ruled_out/](../ruled_out/).
