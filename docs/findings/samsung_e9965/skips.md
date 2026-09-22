# SKIP & CRASH reasons — Samsung ENN / E9965

Per `analysis_single.md` §4–5, the **reason** is the finding for these modes, not just the count.

## SKIP — delegated ops rejected at runtime (coverage gaps)
**3,761** delegated (`ops≥1`) graphs across **31 operators** all fail with the **same** runtime error:

```
executor ExecutorchRuntimeException: [ExecuTorch Error 0x1] Internal error:
Execution failed for method: forward
```

The op passes ENN partitioning (it delegated) but the compiled model cannot execute on the NPU.
The per-op cause is only visible in device logcat (the feed detail is this generic string). Top
operators by SKIP count:

| count | operator | runtime reason |
|--:|---|---|
| 438 | `replication_pad1d.out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 436 | `replication_pad2d.out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 433 | `replication_pad3d.out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 327 | `pixel_shuffle` | [Error 0x1] Internal error: Execution failed for method: forward |
| 237 | `hardtanh.out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 165 | `clamp.out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 147 | `pixel_unshuffle` | [Error 0x1] Internal error: Execution failed for method: forward |
| 125 | `hardtanh` | [Error 0x1] Internal error: Execution failed for method: forward |
| 111 | `bitwise_and.Tensor_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 110 | `bitwise_xor.Tensor_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 107 | `bitwise_or.Tensor_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 105 | `bitwise_or.Scalar_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 97 | `bitwise_xor.Scalar_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 97 | `logit.out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 91 | `bitwise_and.Scalar_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 79 | `relu` | [Error 0x1] Internal error: Execution failed for method: forward |
| 78 | `floor_divide.out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 76 | `minimum.out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 64 | `sqrt.out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 55 | `bitwise_right_shift.Tensor_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 48 | `prod.int_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 47 | `bitwise_left_shift.Tensor_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 46 | `sub.out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 37 | `remainder.Tensor_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 33 | `fmod.Tensor_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 32 | `remainder.Scalar_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 30 | `floor_divide` | [Error 0x1] Internal error: Execution failed for method: forward |
| 20 | `div.out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 17 | `pow.Scalar_out` | [Error 0x1] Internal error: Execution failed for method: forward |
| 14 | `stack` | [Error 0x1] Internal error: Execution failed for method: forward |

_total delegated SKIP: 3761 across 43 operators; all share the one reason above._


Distinct classes worth noting:
- **`replication_pad{1,2,3}d.out`** (~1,300 combined) — every replication-pad form delegated but
  never executes: a systematic ENN pad coverage gap.
- **`pixel_shuffle` / `pixel_unshuffle`** — high SKIP alongside their confirmed MISMATCHes (the forms
  that *do* run are wrong; many others SKIP).
- **`hardtanh.out` / `clamp.out`** — mostly SKIP, some confirmed MISMATCH.
- **bitwise `.Scalar_out`** (`and/or/xor`) — mostly SKIP; their few MISMATCHes are intermittent.

A second SKIP class, `[Error 0x14] Operator missing`, appears only for **portable** (`ops=0`) ops the
device has no kernel for (`_fft_r2c`, `_cdist_forward`, …) — a portable coverage gap, not ENN; not
tabulated here.

## CRASH — all portable-fallback, none from ENN
All **85** CRASHes are `ops=0` (the op fell back to the portable CPU kernel, which native-aborted —
the ENN delegate never ran them). **ENN produced zero crashes.** Native aborts carry no catchable
message (`executor process died (native abort)`); the missing input guard is the portable-kernel bug.

| count | operator | delegation | trigger / note |
|--:|---|---|---|
| 78 | `max_pool2d_with_indices_backward.grad_input` | portable (ops=0) | native abort; backward op, ENN doesn't delegate it |
| 3 | `unfold_copy` | portable (ops=0) | native abort |
| 3 | `narrow_copy.out` | portable (ops=0) | native abort |
| 1 | `narrow_copy` | portable (ops=0) | native abort |

These belong to the portable CPU kernels, not the Samsung/ENN backend — see
[ruled_out/portable_crashes.md](ruled_out/portable_crashes.md).
