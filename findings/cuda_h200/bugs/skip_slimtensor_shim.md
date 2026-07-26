# SlimTensor shim rejects offset≠0 / non-contiguous / bool inputs — SKIP (coverage gap)

- **Failure mode:** SKIP (device runtime rejects a lowered op) · **delegated** · deterministic
- **Root cause (one shim, three symptoms):** the AOTI CUDA runtime's `SlimTensor` (the
  ExecuTorch↔AOTInductor tensor bridge) only accepts tensors that are **storage-offset 0,
  contiguous, and non-bool**. Any op whose input/output is a view or a bool tensor lowers fine but
  is rejected at `execute()` with `error 0x12`.

## Verbatim reasons + forms
| n | op | reason |
|---:|---|---|
| 389 | `select_copy.int` | `[utils.h:38] SlimTensor storage_offset must be 0, got N` (a `select` view has non-zero offset) |
| 176 | `slice_copy.Tensor` | `[utils.h:45] SlimTensor must be contiguous` (strided/sliced view) |
| 16 | `select_copy.int` | `[utils.h:45] SlimTensor must be contiguous` |
| 188 | `any.dims_out` | `[utils.h:55] dtype mismatch, SlimTensor dtype 11 != ETensor dtype 0` (11 = Bool output) |
| 25 | `any.all_out` | same bool-dtype mismatch |

## Assessment
These are **real coverage gaps** in the CUDA runtime, not numerical bugs: sliced/selected
(non-zero-offset or non-contiguous) tensors and bool-producing ops (`any`) cannot execute. Fixing
requires the `SlimTensor` bridge to handle a storage offset, arbitrary strides, and the Bool dtype
(or the partitioner to decline these forms so they fall back to portable instead of failing at run).

## Reproduce (prints the device SKIP status)
```bash
... .venv-cuda/bin/python findings/cuda_h200/bugs/repro.py w0:33   # storage_offset (select_copy.int)
... .venv-cuda/bin/python findings/cuda_h200/bugs/repro.py w0:11   # non-contiguous (slice_copy.Tensor)
... .venv-cuda/bin/python findings/cuda_h200/bugs/repro.py w0:400  # bool dtype (any.dims_out)
```
