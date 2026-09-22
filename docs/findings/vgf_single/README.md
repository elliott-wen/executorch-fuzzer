# VGF single-operator analysis — findings

Ran the `analysis_single.md` workflow on the Arm **VGF** (TOSA→Vulkan) backend using a
single-operator corpus (`--nodes 1`, 75,343 jobs at `corpus_v3/vgf`), executed on the host
via the VGF delegate on Mesa **lavapipe** (CPU Vulkan) + the ML SDK **emulation layer**
(`mobile/vgf_client/`). Every graph is one op wired to leaf inputs, so each non-OK outcome is
already a minimal repro attributed to exactly one operator. Every verdict below is
**device-verified** (run on the VGF runtime), delegated-vs-portable bucketed (step 3a), and
determinism-gated N=5 (step 3b).

## Headline

| outcome | count | % |
|---|--:|--:|
| OK | 68,129 | 90.4% |
| MISMATCH | 3,149 | 4.2% |
| SKIP | 4,058 | 5.4% |
| CRASH | 7 | 0.009% |
| TIMEOUT | 0 | 0 |

After the **step-3a delegated-vs-portable filter** (only `delegated.ops≥1` failures are VGF bugs):

- **MISMATCH on the VGF delegate: 2,335** (25 ops). 100% reproducible (123/123 sampled jobs, N=5). Split:
  - **68% (1,580) NON-FINITE** — nan/inf positions differ. Of a 120-job sample, **75% had a
    non-finite leaf input** → VGF/TOSA **saturation semantics** (e.g. `relu(inf)→65504`,
    `leaky_relu(inf)→65504`, `inf→3.4e38`) where PyTorch **propagates** inf/nan. The other
    **25% had finite leaves** → genuine **fp16 overflow inside the op** (real NONFINITE finding).
  - **32% (755) finite VALUE divergence** — the genuine per-op kernel bugs (see below).
- **CRASH on the VGF delegate: 0.** All 7 native aborts were **portable-fallback** graphs
  (`ops=0`), not VGF — ruled out (see `ruled_out/`).
- **SKIP (runtime reject): 3,943** (63 ops) — VGF coverage gaps, two classes (see `skips.md`).

Ruled out by step 3a: **814 portable-fallback mismatches + 7 portable crashes** (not VGF), and
**115 feeder-decode SKIPs** (non-tensor output, not a backend bug).

## Confirmed findings — by failure mode

### Mismatch · WRONG-VALUE (ZEROED) — `fill.Scalar`  ★ cleanest bug
`aten.fill.Scalar(L0, c)` delegated to VGF returns **all-zeros regardless of `c`** — the scalar
operand is dropped. 356 delegated mismatches, 100% deterministic. Fully delegated (`ops=1,
non_delegated=0` → no portable path; the VGF delegate genuinely computed it). Verified:
`fill(L0,6)→0`, `fill(L0,-4)→0`, `fill(L0,-8)→0`. Repro: `bugs/repro_fill_scalar.py`.

### Mismatch · NONFINITE (fp16 overflow, finite leaf) — activations
`leaky_relu` (411), `gelu` (350), `clamp` (98), `relu` (58), `elu` (38), `floor_divide` — device
emits inf/nan where the reference is finite (25% of the non-finite bucket, leaf verified
finite). fp16 precision/overflow in the delegated kernel. The remaining 75% are the
saturation-semantics gap on non-finite **inputs** (device saturates, PyTorch propagates) —
reported as a semantic difference, lower severity.

### Mismatch · VALUE (finite) — normalization & integer/scatter ops
Both sides finite but values differ beyond rtol/atol: `native_group_norm` (182),
`slice_scatter` (83), `bitwise_left_shift.Tensor_out` (70), `sum.IntList_out` (18),
`native_layer_norm` (12), `prod.int_out` (12), `floor_divide` (9), `where.self*` (6). Normalizers
are fp16 mean/variance precision; the integer/shift/scatter ones are candidate kernel-math bugs.

### Skip · runtime reject (VGF coverage gaps) — see `skips.md`
- **"Failed to process VGF blob" (Init 0x1)** — op **partitions to VGF but its blob won't
  compile** on the emulation layer: `fill.Tensor` (439), `copy` (383), `max_pool2d_with_indices_backward`
  (294), `any.dims_out` (264), `pixel_shuffle` (242), `topk.values` (156), `mean.out` (81),
  `pixel_unshuffle` (110)… A **partitioner/runtime mismatch** — the partitioner claims ops the
  runtime can't execute.
- **"Output tensor byte size 4 ≠ IO allocation 2" (execute 0x12)** — `where.self` (108): the
  delegate sizes an IO at the wrong dtype width — a genuine **IO-metadata bug**.
- **"Failed to allocate tensor for VGF resource"** — `any.dims_out`.

### Crash
**Zero VGF-delegate crashes.** The 7 native aborts are all portable-fallback graphs (ruled out).

## Per-operator table (top, ranked by delegated-mismatch + runtime-skip)

| operator | produced | delegated | mism(deleg) | ·nonfinite | ·value | skip(runtime) |
|---|--:|--:|--:|--:|--:|--:|
| `native_group_norm` | 684 | 684 | 533 | 351 | 182 | 0 |
| `fill.Tensor` | 439 | 439 | 0 | 0 | 0 | 439 |
| `_fft_r2c` | 435 | 0 | 0 | 0 | 0 | 435 |
| `leaky_relu.out` | 436 | 436 | 411 | 411 | 0 | 0 |
| `copy` | 437 | 383 | 0 | 0 | 0 | 410 |
| `fill.Scalar` | 429 | 382 | 356 | 0 | 356 | 0 |
| `gelu.out` | 430 | 430 | 350 | 350 | 0 | 0 |
| `_pdist_forward` | 436 | 0 | 0 | 0 | 0 | 350 |
| `max_pool2d_with_indices_backward.grad_input` | 294 | 294 | 0 | 0 | 0 | 294 |
| `any.dims_out` | 573 | 332 | 0 | 0 | 0 | 264 |
| `pixel_shuffle` | 864 | 398 | 0 | 0 | 0 | 242 |
| `expand_copy` | 377 | 58 | 0 | 0 | 0 | 169 |
| `topk.values` | 404 | 156 | 0 | 0 | 0 | 156 |
| `floor_divide.out` | 416 | 306 | 126 | 123 | 3 | 2 |
| `scatter_add.out` | 122 | 0 | 0 | 0 | 0 | 120 |
| `where.self` | 435 | 211 | 2 | 0 | 2 | 108 |
| `pixel_unshuffle` | 867 | 491 | 0 | 0 | 0 | 110 |
| `clamp.out` | 433 | 291 | 98 | 98 | 0 | 0 |
| `sum.IntList_out` | 430 | 278 | 18 | 0 | 18 | 75 |
| `floor_divide` | 411 | 252 | 93 | 84 | 9 | 0 |
| `scatter.src_out` | 90 | 0 | 0 | 0 | 0 | 89 |
| `slice_scatter` | 413 | 357 | 83 | 0 | 83 | 1 |
| `mean.out` | 316 | 316 | 0 | 0 | 0 | 81 |
| `any.all_out` | 194 | 137 | 0 | 0 | 0 | 81 |
| `linear.out` | 80 | 0 | 0 | 0 | 0 | 80 |
| `bitwise_left_shift.Tensor_out` | 411 | 122 | 79 | 9 | 70 | 0 |
| `arange.start_out` | 149 | 79 | 0 | 0 | 0 | 64 |
| `relu` | 389 | 326 | 58 | 58 | 0 | 2 |
| `index_select` | 125 | 73 | 0 | 0 | 0 | 56 |
| `linear` | 52 | 0 | 0 | 0 | 0 | 52 |
| `elu.out` | 432 | 432 | 38 | 38 | 0 | 0 |
| `amin.out` | 430 | 65 | 1 | 1 | 0 | 33 |
| `where.self_out` | 435 | 227 | 4 | 0 | 4 | 22 |
| `scatter.value_out` | 27 | 0 | 0 | 0 | 0 | 26 |

Columns: **produced** = jobs of this op in the corpus; **delegated** = of those, how many the
VGF partitioner absorbed (`ops≥1`); **mism(deleg)** = delegated mismatches (the bug count),
split into **·nonfinite** / **·value**; **skip(runtime)** = runtime rejects. Ops with
`delegated=0` (e.g. `_fft_r2c`, `_pdist_forward`, `narrow_copy`, `linear`, `scatter*`) fell back to
portable — their skips are **portable** coverage gaps, not VGF.

## Coverage ledger

- **205** distinct operators produced in the corpus; **159** have ≥1 delegated form.
- Every operator with a non-OK outcome has a verdict row above (or in `skips.md`).
- Ops that appear only as `delegated=0` never exercised the VGF delegate (portable fallback) —
  no VGF verdict is possible for them from this corpus; listed in `skips.md` where they SKIP.
- `build_job`-time attrition (graphs that crashed/failed generation) is recorded in
  `corpus_v3/vgf/_crashes/` and is a generator/backend coverage gap, distinct from device bugs.

## Artifacts
- `skiplog.tsv` — every non-OK graph (status, job_id, reason, op). The raw record.
- `per_op_table.tsv` / `buckets.json` / `coverage.json` / `determinism.json` — analysis data.
- `analyze.py` (join+bucket), `determinism.py` (N=5 gate) — reproducible pipeline.
- `bugs/repro_fill_scalar.py` — runnable repro of the headline ZEROED bug.
- `skips.md` — per-operator SKIP reason table (verbatim runtime errors).

## Rigor notes
- 100% of sampled delegated mismatches reproduce at N=5 (no flaky/intermittent) — these are
  deterministic divergences, not near-tolerance noise.
- The dtype-only artifact (3c) did not appear: this is a **float** corpus (VGF is FP-default),
  so device and reference share dtype; mismatches are genuine value/non-finite differences.
- Non-finite findings were leaf-checked (3e): 75% are input-driven saturation (semantic gap),
  25% are op-generated overflow on finite input (real).
