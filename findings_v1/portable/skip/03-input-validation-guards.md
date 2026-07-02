# SKIP reason: shape / argument validation guards

**Count: 10,992 (29.1% of skips).** The kernel ran a shape/rank/dim/size precondition check
and **correctly rejected a malformed input** the fuzzer produced (wrong rank, out-of-range
dim, non-positive size, non-extractable scalar). **Working as intended** — this is exactly the
graceful behavior the [crashing ops](../crash/) are missing.

## Clusters

| count | operator | firing file | guard |
|---:|---|---|---|
| 3,839 | `native_group_norm.out` | `op_native_group_norm.cpp` | `in.size(0) == N` |
| 1,693 | `cumsum.out` | `op_cumsum.cpp` | `dim` in `[-rank, rank)` |
| 1,180 | `_pdist_forward.out` | `op_pdist_forward.cpp` | input must be rank 2 (got 1) |
| 785 | `_adaptive_avg_pool2d.out` | `kernel_ops_util.cpp` | `output_size > 0` (got (0,0)) |
| 753 | `_pdist_forward.out` | `op_pdist_forward.cpp` | input must be rank 2 (got 5) |
| 613 | `native_group_norm.out` | `op_native_group_norm.cpp` | `in.size(1) == C` |
| 592 | `convolution.out` | `op_convolution.cpp` | input rank check |
| 554 | `arange.start_out` | `op_arange.cpp` | `extract_scalar(start)` |
| 532 | `_pdist_forward.out` | `op_pdist_forward.cpp` | input must be rank 2 (got 3) |
| 400 | `_pdist_forward.out` | `op_pdist_forward.cpp` | input must be rank 2 (got 4) |
| 38 | `arange.start_out` | `op_arange.cpp` | `extract_scalar(end)` |

## Notes

- **`native_group_norm` size checks (4,452).** N/C must match the input — the fuzzer feeds
  mismatched group/channel shapes, correctly rejected. (Contrast: when the *args do* pass the
  guard but the variance is degenerate, group_norm produces a **non-finite born-here bug** —
  see [../bugs/normalization-bornhere.md](../bugs/normalization-bornhere.md).)
- **`_pdist_forward` rank!=2 (2,865).** pdist requires a 2-D input; all other ranks rejected.
- **`cumsum dim` out of range (1,693), `arange` non-extractable scalar (592),
  `adaptive_avg_pool2d output_size>0` (785), `convolution` rank.** All standard malformed-arg
  rejections.

## Classification
Working-as-intended input validation. **No action** — these demonstrate the *correct* pattern
the crashing kernels (`narrow_copy`, `unfold_copy`) should follow: validate before dereferencing
`size(dim)`. Useful as the positive control set.
