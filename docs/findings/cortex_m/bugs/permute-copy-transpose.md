# cortex-m `permute_copy` → `cortex_m::transpose` — WRONG-VALUE

- **Failure mode:** MISMATCH (silent wrong output — device RAN, status OK, values wrong)
- **Mechanism:** WRONG-VALUE — output is the **axis-swapped** tensor instead of the requested permutation
- **Kernel bucket:** CM-COMPUTE — the op-under-test ran as a `cortex_m::transpose` CMSIS-NN kernel
  (filter 3a: re-lowering the `.pte` shows `cortex_m::transpose.out` present)
- **Determinism:** REPRO 5/5 on the Corstone-300 FVP (all instances) — deterministic, not flaky
- **Inputs:** finite (filter 3e clean); the divergence is the kernel, not the input
- **Repro:** [repro_permute-copy-transpose.py](repro_permute-copy-transpose.py) — `JOB_ID=w116:752`

## What happens
The CortexM lowering rewrites `aten.permute_copy` into `cortex_m::transpose`. But
`cortex_m::transpose` performs a fixed **axis swap**, not the general N-D permutation the op
requests. When the requested permutation is **identity** (`[0,1]` for a 2-D tensor, `[0,1,2,3]`
for 4-D) — where the correct output is exactly the input — the device still swaps axes and
returns a scrambled tensor.

## Device-verified evidence (job `w116:752`, `permute_copy(L0, [0,1])`, identity)
```
L0 (4,3) fp32     = [-0.480, 1.110, 0.934, 1.164, 0.514, 0.675, 0.103, -0.137, -0.793, -0.913, -0.279, 0.789]
eager (identity)  = [-0.480, 1.107, 0.936, 1.164, 0.513, 0.676, 0.106, -0.138, ...]   # == input, row-major
device (cortex-m) = [-0.480, 1.164, 0.106, -0.912, 1.107, 0.513, -0.138, -0.277, ...] # == input^T, column-major
DEVICE status=MISMATCH  out[0] max|delta|=2.075  (rtol=0.01 atol=0.001)
```
`device[1] = 1.164 = input[3]` (the element at row 1, col 0) — i.e. the device emitted the
**transpose** of L0 where the graph asked for the identity. Same shape `(4,3)` in and out (a
transpose of a shape with unequal dims would change the shape and be caught earlier; the identity
permutation keeps the shape, so the bug is *silent* — wrong values, right shape).

## Prevalence
All **8** `permute_copy` MISMATCH samples in the corpus are this bug (8/8 of permute_copy's
non-OK; the op's other samples are SKIP via the rank-limit guard below). Every classified
CM-COMPUTE MISMATCH in the entire corpus is this one operator — it is the sole cortex-m
compute-kernel value bug found.

## Related (same `cortex_m::transpose` kernel, SKIP not MISMATCH)
`cortex_m::transpose` also **rejects rank ≥ 5** (`op_transpose.cpp:44 transpose_out: expected
tensor rank in [1, 4], got 5/7` → `KernelCall failed … cortex_m::transpose.out: 0x12`). This
makes `pixel_shuffle` (322), `pixel_unshuffle` (313), rank-5 `transpose_copy`/`permute_copy`
SKIP rather than run. See [../skips.md](../skips.md). So `permute_copy`/transpose routes through
`cortex_m::transpose` for every rank ≤ 4 identity/permute and mis-computes it.

## Fix direction
The CortexM permute→transpose rewrite must either (a) only fire for permutations that are an
actual single-axis-pair swap the kernel implements, leaving general permutations (incl. identity)
to the portable `permute_copy` kernel, or (b) make `cortex_m::transpose` honour the full
permutation vector. Today it swaps unconditionally.
