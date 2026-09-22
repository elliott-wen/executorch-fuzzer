# CUDA backend — SKIP & CRASH reason tables

The reason **is** the finding for these modes (a bare count is not). All reasons verbatim from the
runtime; all forms are single-operator, whole-graph-delegated. 971 SKIP, 469 CRASH total.

## SKIP — device runtime rejects a lowered op (real coverage gaps)

The AOTI CUDA runtime's `SlimTensor` shim (the ExecuTorch↔AOTInductor tensor bridge) refuses input
forms that pass partitioning + AOTI compile but cannot execute. **832 of 971 SKIPs are genuine**
runtime rejections; the other 139 are a compile-cache artifact (see bottom / ruled_out).

| count | op(s) | verbatim runtime reason |
|---:|---|---|
| 389 | `select_copy.int` | `client RuntimeError: method->execute() failed with error 0x12 \| [utils.h:38] SlimTensor storage_offset must be 0, got N` — a `select` produces a view with non-zero storage offset; the shim requires offset 0. |
| 176 | `slice_copy.Tensor` | `... \| [utils.h:45] SlimTensor must be contiguous` — a strided/sliced view; the shim requires contiguous inputs. |
| 16 | `select_copy.int` | `[utils.h:45] SlimTensor must be contiguous` (same op, non-contiguous form). |
| 188 | `any.dims_out` | `... \| [utils.h:55] dtype mismatch, SlimTensor dtype 11 != ETensor dtype 0` — dtype 11 = **Bool**; `any` produces a bool output the shim can't reconcile (ETensor dtype 0 = Byte/uninit). |
| 25 | `any.all_out` | same bool-dtype mismatch (`11 != 0`). |
| 38 | `max_pool2d_with_indices_backward.grad_input` | `... error 0x1 \| Assertion 'index out of bounds: 0 <= tmp0 < N' failed` — device-side assert from an out-of-range `indices` input (see CRASH below; SKIP vs CRASH depends on whether the assert is caught before the async abort). |

**Mechanism:** three of these (`storage_offset`, `contiguous`, bool-dtype) are a single root cause —
the `SlimTensor` bridge only accepts **offset-0, contiguous, non-bool** tensors, so any op whose
input/output is a view or a bool tensor is rejected at execute time even though it lowered. This is
a real coverage limitation of the CUDA runtime, not a numerical bug.

## CRASH — native abort (no catchable message unless the assert prints first)

469 total. Two ops dominate (420); a 31-op tail (49) is sporadic edge-form aborts.

| count | op | trigger / captured native error |
|---:|---|---|
| 243 | `max_pool2d_with_indices_backward.grad_input` | Device-side assert `index out of bounds: 0 <= tmp0 < N` (captured by running directly). The backward scatters into `grad_input` using the `indices` input; fuzzer-arbitrary indices are out of range and the Triton kernel asserts → async CUDA abort. **Missing input validation** — should reject with a catchable error, not device-abort. |
| 177 | `clamp.Tensor_out` | `[cuda_allocator.cpp:248] cudaMemcpyAsync failed: an illegal memory access was encountered (48 bytes)` → `[storage.h:158] memcpy_async failed`. A broadcasting form of tensor-min/max clamp drives a GPU out-of-bounds access. |
| 8 | `stack` | native abort (no message) |
| 3 ea | `logical_xor.out`, `logical_and.out`, `copy` | native abort |
| 2 ea | `scatter.src_out`, `rsqrt.out`, `narrow_copy`, `mean.out`, `erf.out` | native abort |
| 1 ea | `var_mean.correction`, `unsqueeze_copy`, `unfold_copy`, `t_copy`, `_softmax.out`, `sinh.out`, `round.out`, `replication_pad3d.out`, `prod.out`, … (18 ops) | native abort (no catchable message) |

A native abort has **no catchable message** — the worker reports `executor died (native abort)` and
the crash-isolated coordinator respawns. The two dominant crashers were reproduced directly (in
`.venv-cuda`) to capture the underlying CUDA error above; the long tail was left as `native abort`.

## Ruled out (not device bugs) — see [ruled_out/compile_cache_loadfail.md](ruled_out/compile_cache_loadfail.md)

| count | class | verbatim |
|---:|---|---|
| 99 | load: undefined symbol (~71 ops) | `Failed loading symbol AOTInductorModelContainerCreateWithDevice ... undefined symbol` — torchinductor compile-cache race; re-lowering runs clean. |
| 40 | load: init failed (~25 ops) | `Init failed for backend CudaBackend: 0x1` — same artifact. |
