| count | operator | representative verbatim device reason (rep job) |
|------:|----------|---------|
| 430 | `stack` | no out-*.bin produced — no catchable device message _(w0:265)_ |
| 389 | `pixel_shuffle` | F [executorch:arm_executor_runner.cpp:1240 run_model()] In function run_model(), assert failed (status == Error::Ok): Execution of method forward failed with st _(w0:167)_ |
| 368 | `pixel_unshuffle` | F [executorch:arm_executor_runner.cpp:1240 run_model()] In function run_model(), assert failed (status == Error::Ok): Execution of method forward failed with st _(w0:113)_ |
| 345 | `zeros.out` | no out-*.bin produced — no catchable device message _(w0:47)_ |
| 339 | `ones.out` | no out-*.bin produced — no catchable device message _(w0:104)_ |
| 333 | `full.out` | no out-*.bin produced — no catchable device message _(w0:144)_ |
| 322 | `arange.start_out` | no out-*.bin produced — no catchable device message _(w0:421)_ |
| 304 | `arange.out` | no out-*.bin produced — no catchable device message _(w0:80)_ |
| 282 | `_pdist_forward` | E [executorch:tensor_util.h:617 tensor_is_rank()] Check failed (static_cast<size_t>(t.dim()) == rank): Expected tensor.dim() to be 2, but got 1 _(w0:531)_ |
| 236 | `native_group_norm` | E [executorch:normalization_ops_util.cpp:160 check_group_norm_args()] Check failed (in.size(0) == N): _(w0:756)_ |
| 173 | `copy` | E [executorch:op_copy.cpp:33 copy_out()] Check failed (non_blocking == false): _(w0:264)_ |
| 155 | `expand_copy` | E [executorch:copy_ops_util.cpp:165 check_expand_copy_args()] Check failed (implicit == false): This operator is not implemented for when implicit == true. _(w1:216)_ |
| 152 | `where.self` | F [executorch:op_dequantize_per_tensor.cpp:44 check_dequantize_args()] In function check_dequantize_args(), assert failed (input.scalar_type() == dtype): input. _(w0:228)_ |
| 115 | `scatter_add.out` | E [executorch:index_util.cpp:156 check_scatter_add_args()] Check failed (index.scalar_type() == ScalarType::Long): Expected dtype int64 for index; index.scalar_ _(w0:624)_ |
| 97 | `topk.values` | F [executorch:arm_executor_runner.cpp:809 runner_init()] In function runner_init(), assert failed (status == Error::Ok): Failed to prepare inputs 0x12 _(w0:490)_ |
| 85 | `scatter.src_out` | E [executorch:index_util.cpp:156 check_scatter_add_args()] Check failed (index.scalar_type() == ScalarType::Long): Expected dtype int64 for index; index.scalar_ _(w0:485)_ |
| 68 | `min.dim_min` | F [executorch:arm_executor_runner.cpp:809 runner_init()] In function runner_init(), assert failed (status == Error::Ok): Failed to prepare inputs 0x12 _(w1:90)_ |
| 61 | `permute_copy` | F [executorch:arm_executor_runner.cpp:1240 run_model()] In function run_model(), assert failed (status == Error::Ok): Execution of method forward failed with st _(w10:512)_ |
| 45 | `max.dim_max` | F [executorch:arm_executor_runner.cpp:809 runner_init()] In function runner_init(), assert failed (status == Error::Ok): Failed to prepare inputs 0x12 _(w0:293)_ |
| 34 | `minimum.out` | F [executorch:cortex_m_ops_common.h:48 validate_cmsis_nn_tensor_requirements()] In function validate_cmsis_nn_tensor_requirements(), assert failed (input1.scala _(w10:758)_ |
| 30 | `convolution` | E [executorch:kernel_ops_util.cpp:389 check_convolution_args()] Check failed (in.dim() == 3 \|\| in.dim() == 4): Expect input tensor to be 3-D or 4-D, but got, 5. _(w10:522)_ |
| 30 | `scatter.value_out` | E [executorch:index_util.cpp:24 check_gather_args()] Check failed (index.scalar_type() == ScalarType::Long): Expected dtype int64 for index; index.scalar_type() _(w10:736)_ |
| 25 | `transpose_copy.int` | F [executorch:arm_executor_runner.cpp:1240 run_model()] In function run_model(), assert failed (status == Error::Ok): Execution of method forward failed with st _(w0:388)_ |
| 10 | `mul.Scalar` | F [executorch:cortex_m_ops_common.h:53 validate_cmsis_nn_tensor_requirements()] In function validate_cmsis_nn_tensor_requirements(), assert failed (input2.scala _(w0:85)_ |
| 7 | `gather.out` | E [executorch:index_util.cpp:24 check_gather_args()] Check failed (index.scalar_type() == ScalarType::Long): Expected dtype int64 for index; index.scalar_type() _(w106:188)_ |
| 6 | `narrow_copy` | F [executorch:tensor_impl.h:136 size()] In function size(), assert failed (dim < dim_ && dim >= 0): Dimension out of range (expected to be in range of [0, 0], b _(w104:389)_ |
| 3 | `unfold_copy` | F [executorch:tensor_impl.h:136 size()] In function size(), assert failed (dim < dim_ && dim >= 0): Dimension out of range (expected to be in range of [0, -1], _(w101:513)_ |
| 2 | `cumsum.out` | E [executorch:tensor_util.h:411 dim_is_valid()] Check failed (dim >= -upper_bound && dim < upper_bound): Dimension 0 is out of range. Dimension should be betwee _(w117:535)_ |
| 2 | `_adaptive_avg_pool2d` | E [executorch:kernel_ops_util.cpp:285 check_adaptive_avg_pool2d_args()] Check failed (output_size[0] > 0 && output_size[1] > 0): output_size must be positive, b _(w124:559)_ |
| 1 | `mm.out` | F [executorch:arm_executor_runner.cpp:1240 run_model()] In function run_model(), assert failed (status == Error::Ok): Execution of method forward failed with st _(w122:545)_ |
| 1 | `max_pool2d_with_indices_backward.grad_input` | F [executorch:arm_executor_runner.cpp:809 runner_init()] In function runner_init(), assert failed (status == Error::Ok): Failed to prepare inputs 0x12 _(w78:568)_ |