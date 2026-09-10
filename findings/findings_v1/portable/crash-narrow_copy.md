# CRASH cluster: narrow_copy (3,758 crashes)

## Summary
The ExecuTorch portable `narrow_copy_out` kernel aborts (SIGABRT, exit 134) whenever it
is given a **negative `dim`** (overwhelmingly `dim=-1`) or a `dim` that is out of range
for the input rank. Its argument-validation helper `check_narrow_copy_args` calls
`in.size(dim)` with the *raw, un-normalized* `dim` before `dim` is converted to a
non-negative index. `TensorImpl::size()` hard-asserts (`ET_CHECK_MSG`) that `dim >= 0 &&
dim < dim_`, so the negative/out-of-range `dim` trips a fatal assert inside what was
supposed to be a graceful `ET_KERNEL_CHECK`. Eager PyTorch handles the identical call
fine (or raises a clean Python `RuntimeError`), so this is a real ExecuTorch portable
bug: a validation function that aborts instead of returning `false`.

## Reproduction
Command (one process per job so the native abort is isolated):
```
cd /data/jwen929/mobile && .venv/bin/python tmp/run_portable/repro_one.py corpus/portable/w0/<job>.job 2>&1; echo "exit=$?"
```

All representative jobs abort at the **same site**, `tensor_impl.h:136` inside
`TensorImpl::size()`, with exit code **134 (SIGABRT)**:

| job_id   | offending narrow_copy call (from the .py)        | input rank | abort message (range / got)                                   |
|----------|--------------------------------------------------|-----------|----------------------------------------------------------------|
| w0:1693  | `narrow_copy.default(n12, -1, 0, 0)`             | 1-D `(2,)`| `expected to be in range of [0, 0], but got -1`                |
| w0:1697  | `narrow_copy.default(n0, -1, 0, 0)`              | 1-D `(1,)`| `expected to be in range of [0, 0], but got -1`                |
| w0:1159  | `narrow_copy.default(n0, -1, 0, 0)`              | 1-D `(4,)`| `expected to be in range of [0, 0], but got -1`                |
| w0:1334  | `narrow_copy.default(..., -1, 0, 0)`             | 1-D       | `expected to be in range of [0, 0], but got -1`                |
| w0:1395  | `narrow_copy.default(..., -1, 0, 0)`             | 1-D       | `expected to be in range of [0, 0], but got -1`                |
| w0:1391  | `narrow_copy.default(..., -1, 0, 0)`             | 1-D       | `expected to be in range of [0, 0], but got -1`                |
| w0:102   | `narrow_copy.default(n0, -1, 0, 0)`              | 1-D `(2,)`| `expected to be in range of [0, 0], but got -1`                |
| w0:1442  | `narrow_copy.default(..., -1, 0, 0)`             | 1-D       | `expected to be in range of [0, 0], but got -1`                |
| w0:104   | `narrow_copy.default(n3, 0, 0, 0)`               | 0-D       | `expected to be in range of [0, -1], but got 0` (variant)      |

Verbatim abort line (w0:1693, identical for the whole `-1` group):
```
[tensor_impl.h:136] In function size(), assert failed (dim < dim_ && dim >= 0): Dimension out of range (expected to be in range of [0, 0], but got -1
```
The range `[0, 0]` is `dim_ - 1`, i.e. the input is a **1-D** tensor (`dim_ == 1`); the
assert fires solely because the raw `dim = -1` is `< 0`. w0:104 is the same bug via a
different path — a **0-D** input (`dim_ == 0`, range `[0, -1]`) given `dim = 0`, where
`0 < dim_` is false.

## Root cause
Call path: `narrow_copy_out` -> `ET_KERNEL_CHECK(check_narrow_copy_args(...))`.

`kernels/portable/cpu/op_narrow_copy.cpp:28-36`
```cpp
  ET_KERNEL_CHECK(
      ctx,
      check_narrow_copy_args(in, dim, start, length, out),   // <-- validates with raw dim
      InvalidArgument,
      out);

  if (dim < 0) {
    dim += in.dim();                                          // <-- normalization happens AFTER the check
  }
```

`kernels/portable/cpu/util/slice_util.cpp:20-37` (`check_narrow_copy_args`)
```cpp
  ET_LOG_AND_RETURN_IF_FALSE(in.dim() > 0);
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));       // tensor_has_dim ACCEPTS dim == -1
  ET_CHECK_OR_RETURN_FALSE(length >= 0, ...);
  ET_LOG_AND_RETURN_IF_FALSE(start >= -in.size(dim));        // line 31: in.size(-1)  -> HARD ABORT
  ET_LOG_AND_RETURN_IF_FALSE(start <= in.size(dim));         // line 32: in.size(-1)
  if (start < 0) { start += in.size(dim); }
  ET_LOG_AND_RETURN_IF_FALSE(start + length <= in.size(dim));
```

`tensor_has_dim` (`runtime/core/exec_aten/util/tensor_util.h:650-664`) explicitly *allows*
`d == -1` (and any in-range negative index) and returns `true` without normalizing it.
The very next lines (slice_util.cpp:31-32) then pass that still-negative `dim` straight
into `Tensor::size(dim)`, which is not negative-aware:

`runtime/core/portable_type/tensor_impl.h:135-141`
```cpp
  ssize_t size(ssize_t dim) const {
    ET_CHECK_MSG(
        dim < dim_ && dim >= 0,                               // dim = -1 fails dim >= 0  -> ET_CHECK aborts
        "Dimension out of range (expected to be in range of [0, %zd], but got %zd",
        dim_ - 1, dim);
    return sizes_[dim];
  }
```

`ET_CHECK_MSG` is a fatal assert (calls `abort()`), not a recoverable check, so it kills
the executor fork before `check_narrow_copy_args` can ever return `false`. The intended
graceful `InvalidArgument` path in `ET_KERNEL_CHECK` is therefore unreachable for negative
or out-of-range `dim`.

## Eager comparison
The exact same calls in eager PyTorch never abort:

- **1-D input, `narrow_copy(a, -1, 0, 0)`** (w0:1693 / 1159 / 102 shape): returns cleanly,
  `torch.Size([0])`. Running the full w0_1693 graph in eager succeeds; `n15` is
  `tensor([])`. ExecuTorch aborts on the identical input.
- **0-D input, `narrow_copy(scalar, -1, 0, 0)`**: eager raises a clean, catchable
  `RuntimeError: narrow() cannot be applied to a 0-dim tensor.` ExecuTorch aborts.

So in every case eager either succeeds or raises a normal Python exception; ExecuTorch
turns the same input into an unrecoverable native abort.

## Classification & severity
- **Real ExecuTorch portable bug.** A kernel *validation* helper must never call a
  fatal-asserting accessor (`Tensor::size()`) on the un-normalized `dim` it is meant to be
  validating. The negative-dim case is fully legal in PyTorch/ATen semantics, so this is
  not a generator-only artifact.
- **Minor generator interaction.** The fuzzer emits a lot of `narrow_copy(x, -1, 0, 0)`
  (length-0 narrows on rank-1 tensors) and some 0-D inputs, which is what surfaces the
  bug. These are valid graphs (eager runs them), so the generator is exercising a genuine
  edge case rather than producing illegal IR. No generator fix is required for
  correctness, though it could optionally normalize `dim` before emission to reduce noise.
- **Severity: high for robustness / DoS.** Any portable-backend deployment that runs a
  model containing `narrow_copy` with a negative dim will hard-abort the runtime instead
  of returning an error — an attacker- or data-controlled crash, not a graceful failure.
- **Likely share of the 3,758 crashes:** dominant. In a sample of 40 distinct
  narrow_copy crash jobs, 36 (90%) contain a `narrow_copy(..., -1, ...)` negative-dim
  call, and every representative job aborts at the identical `tensor_impl.h:136` size()
  assert. The remaining few are the same bug via the 0-D / out-of-range-dim variant
  (w0:104). Treat essentially all 3,758 as this single root cause.

## Recommendation
Fix in `check_narrow_copy_args` (`kernels/portable/cpu/util/slice_util.cpp`): normalize
`dim` to a non-negative index *before* any `in.size(dim)` call, and validate the range
without invoking the asserting accessor. For example:

```cpp
bool check_narrow_copy_args(const Tensor& in, int64_t dim, int64_t start,
                            int64_t length, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(in.dim() > 0);
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(in, dim));
  const int64_t d = dim < 0 ? dim + in.dim() : dim;      // normalize FIRST
  ET_LOG_AND_RETURN_IF_FALSE(d >= 0 && d < in.dim());
  ET_CHECK_OR_RETURN_FALSE(length >= 0, "length must be non-negative; length = %" PRId64, length);
  const int64_t dimsize = in.size(d);                    // now d is guaranteed >= 0
  ET_LOG_AND_RETURN_IF_FALSE(start >= -dimsize && start <= dimsize);
  if (start < 0) start += dimsize;
  ET_LOG_AND_RETURN_IF_FALSE(start + length <= dimsize);
  return true;
}
```

This converts the abort into the intended graceful `InvalidArgument` (for truly invalid
args) and makes the legal negative-dim / length-0 case succeed, matching eager. The same
normalize-before-`size()` pattern should be audited in the sibling helpers in
`slice_util.cpp` (e.g. `check_slice_copy_args`, which has the same `tensor_has_dim` +
later `size(dim)` shape) and anywhere else a raw negative `dim` reaches
`TensorImpl::size()`.
