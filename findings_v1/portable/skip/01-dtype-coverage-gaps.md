# SKIP reason: dtype coverage gaps

**Count: 10,483 (27.8% of skips).** The portable kernel's dtype dispatch hit a dtype it does
not implement and returned `Unhandled dtype …` (error `0x12`). Not a bug — the op simply isn't
compiled for that dtype. These are the cleanest "coverage map" entries.

## Clusters

| count | operator | firing file | dtype refused |
|---:|---|---|---|
| 3,456 | `_conj_physical.out` | `op__conj_physical.cpp` | Float |
| 2,887 | `_conj_physical.out` | `op__conj_physical.cpp` | Long |
| 2,779 | `_conj_physical.out` | `op__conj_physical.cpp` | Half |
| 901 | `_conj_physical.out` | `op__conj_physical.cpp` | Bool |
| 152 | `_pdist_forward.out` | `op_pdist_forward.cpp` | Long |
| 58 | `permute_copy.out` | `op_permute_copy.cpp` | Float8_e4m3fn |
| 49 | `_pdist_forward.out` | `op_pdist_forward.cpp` | Bool |
| 24 | `bitwise_or.Scalar_out` | `bitwise_op.h` | Float |
| 20 | `bitwise_left_shift.Tensor_Scalar_out` | `bitwise_op.h` | Float |
| 20 | `bitwise_right_shift.Tensor_Scalar_out` | `bitwise_op.h` | Float |
| 16 | `bitwise_and.Scalar_out` | `bitwise_op.h` | Float |

## Notes

> Source-level companion: [06-complex-dtype-support.md](06-complex-dtype-support.md) — why this
> is so `_conj_physical`-dominated: portable implements complex in only ~10 of 172 kernels, so a
> complex-only op like `_conj_physical` SKIPs on the real inputs the fuzzer feeds it.

- **`_conj_physical` (10,023 = 96% of this category, 27% of ALL skips) — a REAL coverage gap.**
  Eager `torch.conj_physical` is defined for **all** dtypes: on a real/integer/bool tensor it
  is the identity and returns the tensor unchanged. The portable kernel implements **complex
  only** and refuses everything else. So any program eager runs fine (real-dtype
  `conj_physical`, which appears via decompositions and dtype-generic code) **cannot run on
  ExecuTorch portable** — a genuine op-coverage gap, not generator noise. It just happens to be
  the most-sampled instance of it. Fix belongs in ExecuTorch (add the trivial real-dtype
  identity path), not in the generator.
- **`bitwise_*` Unhandled Float** is *correct*: bitwise ops are integer/bool-only; eager also
  rejects float bitwise ops. Working as intended.
- **`permute_copy` Float8** is a genuine niche dtype gap (fp8 not wired through permute).

## Classification
Coverage gaps in the portable backend, surfaced as graceful refusals. The headline one
(**`_conj_physical` on real dtypes**) is a **real gap worth closing in ExecuTorch** — eager
supports it as identity, portable doesn't, so it blocks otherwise-runnable programs. The
`bitwise_* Float` entries are *correct* refusals (eager rejects float bitwise too). The
`Float8`/exotic-dtype paths are genuine but rare niche gaps. None are correctness bugs; the
actionable one is adding the real-dtype `_conj_physical` path to the portable kernel.
