# CUDA (AOTInductor) backend — single-operator corpus analysis (NVIDIA H200)

ExecuTorch CUDA backend (`EXECUTORCH_BUILD_CUDA=ON`, torch 2.12.0+cu130, from-source
executorch 1.4.0a0), run on 7× NVIDIA H200. Corpus: `corpus_v3/cuda`, **83,534 single-operator
graphs** (`--nodes 1`), 200 of 225 scheduled ops produced. Every job is diffed against the CPU
eager fp32 oracle by `net/compare.py`.

**Whole-graph delegation ⇒ no portable-fallback confound.** The AOTI partitioner takes the whole
graph, so **all 83,534 jobs are delegated (`ops≥1`)** — step 3a is a no-op: every non-OK outcome
is the CUDA delegated kernel, never a portable-CPU-kernel divergence.

## Result tally (whole corpus)

| outcome | count | rate |
|---|---:|---:|
| OK | 80,590 | 96.48% |
| MISMATCH | 1,504 | 1.80% |
| CRASH | 469 | 0.56% |
| SKIP | 971 | 1.16% |

Every non-OK sampled for a finding was **determinism-gated (N=5 independent device replays)** —
all reproduced 5/5 (GPU execution of a fixed compiled `.pte` is deterministic; no flaky drops).

## Headline: the confounds filtered (the real work)

Three large classes are **NOT CUDA kernel bugs** and are moved to [`ruled_out/`](ruled_out/):

1. **Multi-output `out=` output-misalignment (304 MISMATCH + topk value cases).** For
   `min.dim`/`max.dim`/`topk.values`, the harness `user_pos` (from the exported program's
   output_specs) points at an output index the *delegate* leaves empty `(0,0,0,0,0)` — while the
   **correct value sits at a different device index**. Proven not-CUDA-specific: the same graph on
   **xnnpack** mis-aligns identically, and on **portable** it aligns correctly. The device computed
   the right answer. See [ruled_out/multioutput_alignment.md](ruled_out/multioutput_alignment.md).
2. **`_native_batch_norm_legit` reference-degenerate (≈306 MISMATCH).** With near-zero variance the
   **CPU reference** blows up to `NaN`/`2e20`; the device stays finite. The reference is the
   degenerate side (step 3d) — not a device bug. Shape cases are the alignment artifact above.
3. **Compile-cache load failures (139 SKIP).** `undefined symbol AOTInductorModelContainerCreateWithDevice`
   / `Init failed for backend CudaBackend` spread thinly over ~90 ops. **Re-lowering the exact graph
   produces a `.pte` that loads and runs cleanly** → the original `.so` blob was a torchinductor
   compile-cache race artifact from the 128-worker parallel pregen, not an op bug. (Infra note: use
   fewer parallel compile workers, or a per-worker `TORCHINDUCTOR_CACHE_DIR`.)

Also filtered: 136 non-finite MISMATCH whose **leaf input was already non-finite** (step 3e).

## Confirmed CUDA-backend findings

### Mismatch (wrong value) — all delegated, single-output, shapes match
| op(s) | n | mechanism | pinned evidence |
|---|---:|---|---|
| `bitwise_right_shift.Tensor_Scalar` (+`.out`) | ~294 | **WRONG-VALUE** shift ≥ bitwidth / negative shift is UB in the CUDA kernel | int64 `x>>s`: eager `-1`, **device `7.21e16`** (w0:196) |
| `bitwise_left_shift.Tensor_Scalar` (+`.out`) | ~274 | same, left shift | eager `-1`, **device `1.34e10`** (w0:237) |
| `prod.int_out` | 20 | **WRONG-VALUE** integer product overflow handled differently | uint8: eager `0`, **device `255`** (w0:226) |
| `sum.IntList_out` | 17 | **WRONG-VALUE** integer sum | int16 scalar: eager `1`, **device `0`** (w102:69) |

Bit-shift is the strongest finding: ~570 gross wrong-value mismatches, one clean mechanism.
Repro: [bugs/bitwise_shift.md](bugs/bitwise_shift.md).

### Mismatch (non-finite) — fp16 overflow, finite leaf
| op(s) | n | mechanism |
|---|---:|---|
| `gelu.out`, `floor_divide(.out)`, `grid_sampler_2d`, `elu.out` | ~134 | **NONFINITE** fp16 overflow: CPU and device disagree on `NaN` vs `Inf` at the overflow point (e.g. gelu fp16: eager `NaN`, device `Inf`) |

Softer than the integer bugs (both sides are non-finite; fp16 overflow semantics), but a real,
deterministic device≠reference divergence. See [bugs/fp16_nonfinite.md](bugs/fp16_nonfinite.md).

### Crash (native abort)
| op | n | trigger / captured native error |
|---|---:|---|
| `max_pool2d_with_indices_backward.grad_input` | 243 (+38 SKIP) | **device-side assert** `index out of bounds: 0 <= tmp0 < N` — out-of-range `indices` input scatters OOB; no input guard (aborts instead of catchable error) |
| `clamp.Tensor_out` | 177 | **illegal memory access** `cudaMemcpyAsync failed: an illegal memory access` → `[storage.h:158] memcpy_async failed` (GPU OOB) |
| 31 other ops (`stack` 8, `logical_xor/and.out` 3ea, `scatter.src_out`, `mean.out`, …) | 49 | sporadic native aborts (1–8 each), edge-form specific |

Repros: [bugs/crash_maxpool2d_backward.md](bugs/crash_maxpool2d_backward.md),
[bugs/crash_clamp_tensor.md](bugs/crash_clamp_tensor.md).

### Skip (device runtime rejects a lowered op) — real coverage gaps
The AOTI CUDA runtime's `SlimTensor` shim rejects input forms that partition + lower fine but
can't execute. Verbatim reasons + op tables in [skips.md](skips.md).

| class | n | ops | verbatim runtime error |
|---|---:|---|---|
| storage_offset ≠ 0 | 389 | `select_copy.int` | `[utils.h:38] SlimTensor storage_offset must be 0, got N` |
| non-contiguous | 192 | `slice_copy.Tensor` (176), `select_copy.int` (16) | `[utils.h:45] SlimTensor must be contiguous` |
| bool dtype | 213 | `any.dims_out` (188), `any.all_out` (25) | `[utils.h:55] dtype mismatch, SlimTensor dtype 11 != ETensor dtype 0` |
| index-OOB assert | 38 | `max_pool2d_with_indices_backward` | `Assertion index out of bounds: 0 <= tmp0 < N failed` |

## Per-operator table (top by non-OK; full: [per_op_table.tsv](per_op_table.tsv))

| operator | samples | OK | MM | CR | SK | verdict |
|---|---:|---:|---:|---:|---:|---|
| `select_copy.int` | 431 | 25 | 0 | 0 | 406 | SKIP: storage_offset / non-contig |
| `_native_batch_norm_legit` | 430 | 88 | 341 | 1 | 0 | **ruled out** (ref-degenerate + alignment) |
| `max_pool2d_with_indices_backward.grad_input` | 293 | 12 | 0 | 243 | 38 | **CRASH** index-OOB |
| `topk.values` | 393 | 165 | 228 | 0 | 0 | **ruled out** (multi-output alignment) |
| `bitwise_right_shift.Tensor_Scalar` (+`.out`) | 861 | ~550 | ~294 | 0 | ~4 | **MISMATCH** wrong-value |
| `bitwise_left_shift.Tensor_Scalar` (+`.out`) | 871 | ~580 | ~274 | 1 | ~2 | **MISMATCH** wrong-value |
| `any.dims_out` | 869 | 678 | 0 | 1 | 190 | **SKIP** bool dtype |
| `clamp.Tensor_out` | 436 | 256 | 0 | 177 | 3 | **CRASH** illegal-mem |
| `slice_copy.Tensor` | 430 | 254 | 0 | 0 | 176 | **SKIP** non-contiguous |
| `min.dim_min` / `max.dim_max` | 751 | ~660 | 87 | 0 | 0 | **ruled out** (multi-output alignment) |
| `gelu.out` | 430 | 365 | 65 | 0 | 0 | **MISMATCH** fp16 non-finite |

## Coverage ledger

- **Scheduled usable ops:** 225 (`scheduled_ops.txt`). **Produced in corpus:** 200. **Verdicted:** 200.
- **Coverage gap — 25 ops scheduled but never lowered** (`coverage_gap.txt`): `build_job` SKIP/crash
  at generation time or seed-infeasible (a generator/backend attrition gap, not a device bug).
- `covered_ops(200) + gap_ops(25) == scheduled_ops(225)` ✓.

## Reproduce

```bash
# build the corpus (memory: cuda-backend-setup):  NODES=1 CONCURRENCY=7 ./run_pregen_cuda.sh 128 100000 corpus_v3/cuda
# run + diff the whole corpus on the H200s:
findings/cuda_h200/feed_cuda.sh                       # broker + 14 GPU clients + feed
# rebuild the tables from the skiplog:
.venv/bin/python findings/cuda_h200/build_opmap.py
.venv/bin/python findings/cuda_h200/build_table.py
# one finding, device-verified (in .venv-cuda):
CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64 \
  PYTHONPATH=/data/jwen929 .venv-cuda/bin/python findings/cuda_h200/bugs/repro.py w0:196
```

## Artifacts
- `skiplog.tsv` — every non-OK job (status, job_id, verbatim reason, op).
- `per_op_table.tsv`, `reason_classes.tsv`, `per_op_jobs.tsv`, `opmap.tsv` — the tables.
- `replay_verify.tsv` — determinism-gate results.
- `bugs/` — confirmed findings + `repro.py`. `skips.md` — SKIP/CRASH reason tables. `ruled_out/` — filtered confounds with evidence.
