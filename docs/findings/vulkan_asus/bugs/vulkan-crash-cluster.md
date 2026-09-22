# Vulkan native-abort crash cluster (copy / view / alias ops)

- **Mode:** CRASH (native abort — no catchable error) · **Occurrences:** ~555 genuine aborts
  (after re-running out the ~139 `executor unavailable` dropped-worker artifacts)

Vulkan crashes far more than xnnpack/portable (~60). The crashing ops are dominated by
**copy/view/alias/reshape** kernels:

| count | operator |
|---:|---|
| 103 | `clone` |
| 96 | `lift_fresh_copy` |
| 86 | `alias_copy` |
| 80 | `max_pool2d_with_indices_backward` |
| 63 | `unfold_copy` |
| 63 | `split_with_sizes_copy` |
| 29 | `expand_copy` |
| 13 | `transpose_copy.int` |

These take a native abort (process dies) rather than a catchable error — the Vulkan copy/view kernels
lack the guards the SKIP-path kernels have. Matches the known Vulkan copy/view crash family.
