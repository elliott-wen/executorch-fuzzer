# cortex-m — SKIP / CRASH / TIMEOUT reason tables

Backend: **cortex-m** (int8 CMSIS-NN operator library on Cortex-M55, no NPU), run on the Arm
**Corstone-300 FVP** via `fvp_client`. Corpus: `corpus_v3/cortex-m` (single-operator, `--nodes 1`,
`quant=ALWAYS`). Method: `analysis_single.md`. The **reason is the finding** for these modes —
each row carries the verbatim runtime message from the device `uart.log` (captured by re-running a
representative job through the runner with a persistent out dir; see `_work/cm_repro.py`).

**How to read the "whose kernel" column.** cortex-m is pass-based, so `delegated.ops` is always 0.
The real signal is which `cortex_m::` kernel the lowered `.pte` calls (`_work/cm_kernels.py`).
A SKIP raised inside a `cortex_m::…` kernel or `cortex_m_ops_common.h` is a **cortex-m** finding; a
SKIP raised inside a portable `executorch:op_*` / `*_util` kernel is a **portable-kernel coverage
gap** (the op fell back to the portable kernel, which rejected the form) — not a cortex-m bug, but a
real gap in what this corpus can exercise on the target. A `runner_init: Failed to prepare inputs`
is an **arm_executor_runner** host-seam limit.

## SKIP — cortex_m kernel guards (in scope)
The `cortex_m::` kernel itself rejects/aborts on the form. `cortex_m::transpose` returns a catchable
`Error 0x12`; the `validate_cmsis_nn_tensor_requirements` dtype checks are **hard `ET_CHECK` aborts**
(fatal `F`), which the runner surfaces as a method failure (SKIP) rather than a graceful reject.

| count | operator | cortex_m kernel | verbatim device reason |
|------:|----------|-----------------|------------------------|
| 389 | `pixel_shuffle` | `cortex_m::transpose` | `op_transpose.cpp:44 transpose_out: expected tensor rank in [1, 4], got 7` → `KernelCall failed … cortex_m::transpose.out: 0x12` |
| 368 | `pixel_unshuffle` | `cortex_m::transpose` | same — rank-7 intermediate rejected by `cortex_m::transpose` |
| 61 | `permute_copy` | `cortex_m::transpose` | `op_transpose.cpp:44 … rank in [1,4], got 5` (rank-5 permutes; the rank≤4 identity/permute case MIS-COMPUTES instead — see [bugs/permute-copy-transpose.md](bugs/permute-copy-transpose.md)) |
| 25 | `transpose_copy.int` | `cortex_m::transpose` | `op_transpose.cpp:44 … rank in [1,4], got 5` |
| 34 | `minimum.out` | `cortex_m::minimum` | `cortex_m_ops_common.h:48 validate_cmsis_nn_tensor_requirements: Input1 dtype must be 1 (int8), got 3 (int32)` — **hard abort** |
| 10 | `mul.Scalar` | `cortex_m::quantized_mul` | `cortex_m_ops_common.h:53 validate_cmsis_nn_tensor_requirements: Input2 dtype must be 1 (int8), got 6 (float32)` — **hard abort** |

**`minimum`/`mul` are a lowering bug, not just a guard:** the CortexM pass rewrote the op into the
int8 `cortex_m::minimum` / `cortex_m::quantized_mul` kernel, but the operands were left `int32` /
`float32` (the quantizer did not annotate them). The kernel then `ET_CHECK`-aborts on the dtype. The
pass should not route to the int8 CMSIS kernel unless both operands are int8.

## SKIP — portable-kernel guards (coverage gaps, not cortex-m bugs)
The compute op fell back to the portable `executorch` kernel (only `cortex_m::quantize/dequantize`
at the boundary, or no cortex_m op), and the **portable** kernel rejected the form. Real coverage
gaps of what this single-op corpus reaches on-device; attribute to portable executorch, not cortex-m.

| count | operator | verbatim portable reason |
|------:|----------|--------------------------|
| 282 | `_pdist_forward` | `tensor_util.h:617 tensor_is_rank: Expected tensor.dim() to be 2, but got 1` |
| 236 | `native_group_norm` | `normalization_ops_util.cpp:160 check_group_norm_args: Check failed (in.size(0) == N)` |
| 173 | `copy` | `op_copy.cpp:33 copy_out: Check failed (non_blocking == false)` |
| 155 | `expand_copy` | `copy_ops_util.cpp:165 check_expand_copy_args: not implemented for implicit == true` |
| 152 | `where.self` | `op_dequantize_per_tensor.cpp:44 check_dequantize_args: Check failed (input.scalar_type() == dtype)` |
| 115 | `scatter_add.out` | `index_util.cpp:156 check_scatter_add_args: Expected dtype int64 for index` |
| 85 | `scatter.src_out` | `index_util.cpp:156 check_scatter_add_args: Expected dtype int64 for index` |
| 30 | `scatter.value_out` | `index_util.cpp:24 check_gather_args: Expected dtype int64 for index` |
| 30 | `convolution` | `kernel_ops_util.cpp:389 check_convolution_args: Expect input tensor to be 3-D or 4-D, but got 5` |
| 7 | `gather.out` | `index_util.cpp:24 check_gather_args: Expected dtype int64 for index` |
| 6 | `narrow_copy` | `tensor_impl.h:136 size(): Dimension out of range` |
| 3 | `unfold_copy` | `tensor_impl.h:136 size(): Dimension out of range` |
| 2 | `cumsum.out` | `tensor_util.h:411 dim_is_valid: Dimension 0 is out of range` |
| 2 | `_adaptive_avg_pool2d` | `kernel_ops_util.cpp:285 check_adaptive_avg_pool2d_args: output_size must be positive` |

## SKIP — arm_executor_runner host-seam (no catchable kernel message)
Either the runner could not prepare inputs, or the graph produced no output tensor and no device
error was printed. Factory ops (`zeros/ones/full/arange`) and `stack` produce **no `out-*.bin`**;
reductions/`topk` fail at `runner_init: Failed to prepare inputs 0x12`. Not cortex-m kernel bugs.

| count | operator | reason |
|------:|----------|--------|
| 430 | `stack` | no `out-*.bin` produced — no catchable device message |
| 345 | `zeros.out` | no `out-*.bin` produced — no catchable device message |
| 339 | `ones.out` | no `out-*.bin` produced — no catchable device message |
| 333 | `full.out` | no `out-*.bin` produced — no catchable device message |
| 322 | `arange.start_out` | no `out-*.bin` produced — no catchable device message |
| 304 | `arange.out` | no `out-*.bin` produced — no catchable device message |
| 97 | `topk.values` | `arm_executor_runner.cpp:809 runner_init: Failed to prepare inputs 0x12` |
| 68 | `min.dim_min` | `runner_init: Failed to prepare inputs 0x12` |
| 45 | `max.dim_max` | `runner_init: Failed to prepare inputs 0x12` |
| 1 | `mm.out` | `run_model: Execution of method forward failed` |
| 1 | `max_pool2d_with_indices_backward.grad_input` | `runner_init: Failed to prepare inputs 0x12` |

## CRASH — all INFRA (runner build failures, NOT device crashes)
Per the rigor reminder, these are **not** device native aborts — the client reports `fvp_runner
rc=5` (build fail) as CRASH. Reproduced deterministically: the executorch arm portable_ops_lib
codegen emits `arm_portable_ops_lib/RegisterCodegenUnboxedKernelsEverything.cpp:15: error: empty
filename in #include` — the selected op has **no portable kernel source** (e.g. `aten::linear.out`
has none; linear normally decomposes to addmm/mm). So the semihosting runner never builds and the
op gets **no device verdict**. Attribute to executorch portable-codegen + the per-pte runner link,
not cortex-m.

| count | operator | cause |
|------:|----------|-------|
| 116 | `linear.out` | runner build rc=5 — empty `#include` (no portable `linear.out` kernel) |
| 114 | `linear` | runner build rc=5 — empty `#include` |
| 23 | `_fft_r2c` | runner build rc=5 — no portable kernel |

**0 genuine device crashes** (no native aborts) were observed across 61,735 graphs.

## TIMEOUT
| count | operator | note |
|------:|----------|------|
| 419 | `_fft_r2c` | runner build/run exceeds the 300 s job budget (same missing-kernel class as its rc=5 crashes) — build/codegen hang, not a device hang |

`linear`, `linear.out`, and `_fft_r2c` therefore have **no device coverage** at all (100 % build
attrition) — a generator/portable-codegen gap distinct from any device behaviour.
