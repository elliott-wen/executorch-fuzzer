# Core ML (Apple-Silicon Mac) — confirmed bug index

Every entry is device-verified and deterministic (5/5 re-runs), delegated to Core ML
(`delegated.ops≥1`), on finite in-distribution inputs. See [../README.md](../README.md) for the
per-operator table, coverage ledger, and the ruled-out confounds.

## MISMATCH — wrong value
| bug | op(s) | mechanism |
|-----|-------|-----------|
| [atan2.md](atan2.md) | `atan2.out` | `atan2(0, x<0) → 0` instead of `π` |
| [remainder-scalar.md](remainder-scalar.md) | `remainder.Scalar` | output ZEROED |
| [mean-int.md](mean-int.md) | `mean` | integer-truncated mean of int input |
| [upsample-bilinear2d-aa.md](upsample-bilinear2d-aa.md) | `_upsample_bilinear2d_aa.out` | antialias ignored (Δ≤2.33) |
| [avg_pool2d.md](avg_pool2d.md) | `avg_pool2d.out` | wrong divisor (~2×) + SKIP + rare crash |
| [bitwise-left-shift.md](bitwise-left-shift.md) | `bitwise_left_shift.Tensor_out` | int shift overflow → `-2³¹` / `-inf` |
| [max_pool2d-backward.md](max_pool2d-backward.md) | `max_pool2d_with_indices_backward.grad_input` | wrong gradient scatter (+ crash form) |
| [precision-reductions.md](precision-reductions.md) | `var.correction`, `var.correction_out`, `_native_batch_norm_legit.no_stats`, `logit`, `pow.Scalar_out`, `remainder.Tensor_out`, `convolution` | precision beyond tolerance |

## CRASH — native abort (no catchable message)
| bug | ops |
|-----|-----|
| [crash-native-abort.md](crash-native-abort.md) | `unfold_copy`, `prod.int_out`, `diagonal_copy`, `scatter.value_out`, `max_pool2d_with_indices_backward.grad_input`, `linear.out`, `mean.out`, `cumsum.out`, `logical_or.out`, `mm.out`, `t_copy`, `tril.out`, `alias_copy`, `view_copy` |

## HANG — reproducible timeout (triaged as crash)
| bug | ops |
|-----|-----|
| [hang-timeout.md](hang-timeout.md) | `slice_scatter`, `select_scatter`, `max.unary_out` |

## SKIP — runtime coverage gaps
See [../skips.md](../skips.md) — a per-operator table of the verbatim runtime reasons for all 3,653
SKIP outcomes (Core ML compile failures, static-tensor resize, unsupported dtypes, etc.).
