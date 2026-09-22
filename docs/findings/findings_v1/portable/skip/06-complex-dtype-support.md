# Coverage finding: portable kernels are real-only — only ~10 ops support complex

**Source-level companion to [01-dtype-coverage-gaps.md](01-dtype-coverage-gaps.md).** Where 01
reports the observed `_conj_physical` "Unhandled dtype" SKIPs, this file documents the broader,
source-confirmed fact behind them: the ExecuTorch **portable** CPU backend implements complex
arithmetic in **only ~10 of its 172 `op_*.cpp` kernels**; every other kernel is real-only and
returns `InvalidArgument` ("Unhandled dtype") on a complex input.

This is **not a correctness bug** — it is a backend capability gap (a coverage map entry). It is
why the differential fuzzer collapses every complex dtype to `float32` before generating a graph
([gen/concretize/dtypes.py](../../../gen/concretize/dtypes.py), `COMPLEX32/64/128 → FLOAT32`):
a complex leaf would SKIP the moment it reaches any of the ~160 real-only compute kernels.

## The complete list of complex-capable portable kernels (10 / 172)

All dispatch through `ET_SWITCH_COMPLEXH_TYPES`
(`runtime/core/exec_aten/util/scalar_type_util.h:1287`), whose enumeration
`ET_FORALL_COMPLEXH_TYPES` (`:363-366`) is exactly **`ComplexHalf`, `ComplexFloat`,
`ComplexDouble`** (`complex<Half|float|double>`):

| kernel | complex behavior |
|---|---|
| `op_add.cpp`, `op_mul.cpp`, `op_div.cpp`, `op_sum.cpp`, `op_bmm.cpp`, `op_cat.cpp` | **dual** — branch on the out dtype: complex → `ET_SWITCH_COMPLEXH_TYPES`, else the real switch (e.g. `op_mul.cpp:62` complex vs `:70` `ET_SWITCH_REALB_TYPES`) |
| `op_abs.cpp` | complex in → real magnitude out |
| `op__to_dim_order_copy.cpp` | complex-capable copy/convert |
| `op__conj_physical.cpp` | **complex-only** — `ET_SWITCH_COMPLEXH_TYPES` with no real branch (`op__conj_physical.cpp:35`) |
| `op_view_as_real_copy.cpp` | **complex-only** — reinterprets `complex<T>` as a trailing-2 real tensor |

Everything else — the other **~162 kernels** — uses a real-only switch
(`ET_SWITCH_REAL_TYPES` / `REALHBBF16` / `INT`, whose enumerations contain no complex type) and so
falls through to the `default:` arm.

## Why a complex input becomes a SKIP (not a crash)

A real-only `ET_SWITCH` on a complex `scalar_type()` reaches the catch-all in `ET_INTERNAL_SWITCH`
(`runtime/core/exec_aten/util/scalar_type_util.h:922-931`):

```c
default:
  CONTEXT.fail(torch::executor::Error::InvalidArgument);
  ET_LOG(Error, "Unhandled dtype %s for %s", toString(_st), et_switch_name);
```

`Error::InvalidArgument` is a returned error, not an abort — the feeder records it as **SKIP**, so
the gap surfaces as a graceful refusal, never a crash. (Contrast the XNNPACK delegate, where an
unhandled `ComplexFloat` reaching a *generic elementwise* path instead **segfaults** —
[../../xnnpack_x64/crash → ComplexFloat](../../xnnpack_x64/crash/README.md) /
[portable crash tail](../crash/07-complexfloat-elementwise.md). On portable it is a clean SKIP.)

## The one real gap inside this: `_conj_physical` on real dtypes

`_conj_physical` is **complex-only** by construction (`op__conj_physical.cpp:35`), but eager
`torch.conj_physical` is defined for **all** dtypes — on a real/integer/bool tensor it is the
**identity**. So a program that runs `conj_physical` on a real tensor (common via decompositions
and dtype-generic code) runs fine in eager but **cannot run on portable**. This is the single
largest SKIP in the whole portable run — **10,023 rows** ([01](01-dtype-coverage-gaps.md):
Float 3,456 + Long 2,887 + Half 2,779 + Bool 901) — and it is a **real, actionable ExecuTorch gap**:
add the trivial real-dtype identity path so `_conj_physical` returns the input unchanged for
non-complex inputs. (Note the irony with the fuzzer: because complex is collapsed to `float32`
upstream, `_conj_physical` only ever *sees* real inputs here, so it SKIPs from the real side — the
collapse cannot avoid it; only the kernel fix can.)

## Classification & recommendation

- **Class:** backend **capability / coverage gap**, surfaced as graceful `Unhandled dtype` SKIPs.
  Not a correctness bug — the kernels honestly refuse what they don't implement.
- **Mostly intentional.** Complex tensors are rare in mobile/edge inference; the chief realistic
  source is `_fft_r2c` / FFT-family ops producing a `ComplexFloat` that a downstream **real-only**
  elementwise op then can't consume. For such graphs portable SKIPs (and the XNNPACK delegate can
  crash — see the cross-links above).
- **Actionable items:**
  1. **`_conj_physical` real-dtype identity path** (the one clear gap; unblocks 10,023 otherwise-
     runnable programs). Eager treats it as identity on non-complex dtypes; portable should too.
  2. If complex support is desired for FFT pipelines, widen the relevant elementwise/reduction
     kernels (`exp`, `mul`, `add`, … ) to `ET_SWITCH_*_AND_COMPLEX` variants — but this is a large,
     low-priority surface for mobile and is **not** recommended wholesale.
  3. For the **fuzzer**, the current `complex → float32` collapse
     ([gen/concretize/dtypes.py](../../../gen/concretize/dtypes.py)) is the right call: it keeps
     graphs in the runnable real-only envelope. The shorthand comment there ("portable kernels are
     real-only") is accurate for ~95% of kernels; the ~10 complex-capable ops above are the
     exception.

### Cited sources
- Complex switch + enumeration: `runtime/core/exec_aten/util/scalar_type_util.h:1287` (`ET_SWITCH_COMPLEXH_TYPES`), `:363-366` (`ET_FORALL_COMPLEXH_TYPES` = ComplexHalf/Float/Double)
- Unhandled-dtype SKIP path: `scalar_type_util.h:922-931` (`ET_INTERNAL_SWITCH` default → `InvalidArgument` + "Unhandled dtype")
- Complex-capable kernels: `kernels/portable/cpu/{op_add,op_mul,op_div,op_sum,op_bmm,op_cat,op_abs,op__to_dim_order_copy,op__conj_physical,op_view_as_real_copy}.cpp` (10 of 172 `op_*.cpp`)
- Dual real/complex branch example: `op_mul.cpp:62` (complex) vs `:70` (real)
- Complex-only example: `op__conj_physical.cpp:35`
- Fuzzer collapse: `gen/concretize/dtypes.py` (`COMPLEX32/64/128 → FLOAT32`)
