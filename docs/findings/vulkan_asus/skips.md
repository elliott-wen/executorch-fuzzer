# SKIP + CRASH reasons per operator (vulkan, ASUS phone)

SKIP = the .pte loaded but the Vulkan runtime rejected it (coverage gap). CRASH = native abort.

## SKIP (top 20; verbatim Vulkan-runtime check)
| count | operator | reason |
|---:|---|---|
| 869 | `native_group_norm` | vulkan/runtime/graph/ops/impl/GroupNorm.cpp:188: (graph.val_is_tref(weight_data)) is false |
| 811 | `native_layer_norm` | vulkan/runtime/graph/ops/impl/NativeLayerNorm.cpp:95: native_layer_norm requires weight to be non-None |
| 445 | `_native_batch_norm_legit_no_training` | vulkan/runtime/graph/ops/impl/BatchNorm.cpp:65: (in_sizes.size() == 4) is false |
| 441 | `_fft_r2c` | Operator missing: Execution failed for method: forward |
| 346 | `_pdist_forward` | executor ExecutorchInvalidArgumentException: [ExecuTorch Error 0x12] Invalid arg |
| 319 | `div.out_mode` | vulkan/runtime/graph/ComputeGraph.h:596: Cannot extract scalar from Value with type NONE |
| 308 | `avg_pool2d.out` | vulkan/runtime/utils/VecUtils.h:367: (ints.size() == 2) is false |
| 162 | `copy` | executor ExecutorchInvalidArgumentException: [ExecuTorch Error 0x12] Invalid arg |
| 157 | `argmin.out` | vulkan/runtime/graph/ops/impl/ArgReduce.cpp:28: (graph.is_buffer_storage(in)) is false |
| 145 | `argmax.out` | vulkan/runtime/graph/ops/impl/ArgReduce.cpp:28: (graph.is_buffer_storage(in)) is false |
| 138 | `convolution` | vulkan/runtime/graph/ops/impl/Staging.cpp:217: (graph.val_is_tref(tensor_data)) is false |
| 130 | `_softmax.out` | vulkan/runtime/utils/VecUtils.h:275: (i >= 0 && i < N) is false |
| 116 | `mean.out` | vulkan/runtime/graph/ops/impl/Reduce.cpp:355: dims_list=None only supported for 1D tensors |
| 110 | `scatter_add.out` | executor ExecutorchInvalidArgumentException: [ExecuTorch Error 0x12] Invalid arg |
| 109 | `_log_softmax.out` | vulkan/runtime/utils/VecUtils.h:275: (i >= 0 && i < N) is false |
| 98 | `eq.Tensor_out` | vulkan/runtime/api/ShaderRegistry.cpp:54: (it |
| 94 | `linear.out` | executor IllegalArgumentException: unsupported input dtype code 24 on Android |
| 88 | `sum.IntList_out` | vulkan/runtime/graph/ops/impl/Reduce.cpp:354: dims_list=None only supported for 1D tensors |
| 76 | `scatter.src_out` | executor ExecutorchInvalidArgumentException: [ExecuTorch Error 0x12] Invalid arg |
| 53 | `topk.values` | vulkan/runtime/graph/ComputeGraph.cpp:981: Unsupported type conversion from Byte to staging dtype Int |

## CRASH — native abort (Vulkan copy/view/alias cluster + others)

Corrected: raw run logged 694 CRASH; ~139 were `executor unavailable` (dropped worker, re-run),
leaving ~555 genuine native aborts. Top crashing ops:

| count | operator |
|---:|---|
| 103 | `clone` |
| 96 | `lift_fresh_copy` |
| 86 | `alias_copy` |
| 80 | `max_pool2d_with_indices_backward.grad_input` |
| 63 | `unfold_copy` |
| 63 | `split_with_sizes_copy` |
| 29 | `expand_copy` |
| 13 | `transpose_copy.int` |
| 9 | `min.dim_min` |
| 7 | `max.dim_max` |
| 3 | `topk.values` |
| 2 | `narrow_copy` |
| 1 | `view_copy` |
