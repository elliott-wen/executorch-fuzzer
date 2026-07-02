# SKIP reason: other guards (hard-disabled paths + out-resize failures)

**Count: 2,990 (7.9% of skips).** Two unrelated sub-groups that don't fit the dtype / shape /
promotion categories.

## A. `_log_softmax` hard-disabled path — `Check failed (false)` (2,872)

| count | operator | firing file | guard |
|---:|---|---|---|
| 2,872 | `_log_softmax.out` | `op_log_softmax.cpp` | `Check failed (false)` |

A bare `ET_KERNEL_CHECK(... false ...)` is a **hard-stubbed branch** — the kernel reaches a
case it deliberately does not implement and unconditionally fails. This is the second-largest
single skip cluster (after `_conj_physical`). The most likely trigger is a half-precision /
unsupported-dtype or unsupported-`dim` branch that the portable `_log_softmax` left
unimplemented and gated behind `Check(false)` rather than a descriptive guard.

**Worth a closer look** — unlike a dtype/shape guard, a `Check(false)` means a whole code path
is missing, and `_log_softmax` is a common op a real model would hit. (Note: when
`_log_softmax` *does* run, it has a separate [born-here non-finite bug](../bugs/normalization-bornhere.md)
on size-1 reduce dims.) Recommend identifying which input condition routes to the `false`
branch and either implementing it or replacing the bare `false` with a descriptive
`InvalidArgument` message.

## B. Output-resize failures — `Check failed (error == Error::Ok)` (~90)

| count | operator | firing file | guard |
|---:|---|---|---|
| 32 | `div.out` | `binary_ops.cpp` | `Failed to resize output tensor` |
| 24 | `sub.out` | `binary_ops.cpp` | `Failed to resize output tensor` |
| 19 | `div.out` | `binary_ops.h` | `Failed to resize output tensor` |
| 15 | `sub.out` | `binary_ops.h` | `Failed to resize output tensor` |

The kernel could not resize the pre-allocated `out=` tensor to the broadcast result shape
(the program supplied an out buffer whose shape can't be resized to match). This is an
internal error propagated up gracefully — related to the static-shape / out-buffer model
discussed in [../mismatch-shape.md](../mismatch-shape.md). Generator-driven (an out buffer
that doesn't fit the broadcast); not a backend correctness bug.

## Classification
- **A (`_log_softmax` Check(false))**: a real unimplemented-path gap worth investigating — the
  most actionable item in this file.
- **B (resize failures)**: graceful internal errors from incompatible `out=` buffers; harness/
  generator-side, not a backend bug.
