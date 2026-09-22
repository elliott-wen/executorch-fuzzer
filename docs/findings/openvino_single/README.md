# OpenVINO single-operator analysis — findings

Ran the [`analysis_single.md`](../../analysis_single.md) workflow on the Intel **OpenVINO** (CPU)
ExecuTorch backend using a single-operator corpus (`--nodes 1`, **82,356 jobs** at
`corpus_v3/openvino`), executed on the **host** via the OpenVINO delegate on the ExecuTorch host
runtime (`openvino_client/`, `OPENVINO_DEVICE="CPU"`). Every graph is one operator wired to leaf
inputs, so each non-OK outcome is already a minimal repro attributed to exactly one operator. Every
verdict below is **device-verified** (run on the OpenVINO runtime), **delegated-vs-portable
bucketed** (step 3a), and **determinism-gated N=5** (step 3b).

## Headline

| outcome | count | % |
|---|--:|--:|
| OK | 65,278 | 79.3% |
| MISMATCH | 6,000 | 7.3% |
| SKIP | 11,009 | 13.4% |
| CRASH | 69 | 0.08% |
| TIMEOUT | 0 | 0 |

After the **step-3a delegated-vs-portable split** (only `delegated.ops≥1` failures are OpenVINO bugs):

- **MISMATCH on the OpenVINO delegate: 5,417** (31 ops). Determinism gate: **235/237** sampled
  failing jobs REPRO on all 5 re-runs (2 intermittent, both the max_pool crash). Ruled out:
  **583 portable-fallback mismatches** (6 ops, step 3a) and — within the delegated set — the
  **`native_group_norm` non-finite bucket is a reference-nan confound** (10/12 sampled: PyTorch
  itself returns nan, device is finite; step 3d/3e).
- **CRASH on the OpenVINO delegate: 58** (4 ops) — native aborts, dominated by
  `max_pool2d_with_indices_backward.grad_input` (55). Ruled out: **11 portable-fallback crashes**
  (`narrow_copy`, not OpenVINO).
- **SKIP: 11,009.** Two classes: **10,143** are the OpenVINO delegate's own blob **execute failures**
  (`error 0x1`, 134 ops) — real OpenVINO coverage gaps; **866** are **portable-kernel** rejections
  (`error 0x12`, 5 ops) — not OpenVINO. See [`skips.md`](skips.md).

## Confirmed findings — by failure mode
Full per-bug docs in [`bugs/`](bugs/INDEX.md); each is device-verified and REPRO N=5.

### Mismatch · ZEROED (dropped store) — copy/view/identity family  ★ headline
`clone`, `alias_copy`, `lift_fresh_copy`, `t_copy`, `transpose_copy.int` delegated to OpenVINO return
an **all-zero output** regardless of input/dtype (float32/float16/bool/int64) — the store into the
output tensor is dropped. Verified e.g. `clone([−0.48,1.11,…])→[0,0,…]`, `alias_copy([−3,−3,−3,4])→
[0,0,0,0]`. Partially ZEROED (shape/dtype-dependent): `reflection_pad1d.out`, `rsub.Scalar`,
`var.correction`, `var.correction_out`. → [`bugs/zeroed_copy_ops.md`](bugs/zeroed_copy_ops.md)

### Mismatch · WRONG-VALUE (view transform ignored)
`diagonal_copy` returns the first **row** instead of the diagonal; `unfold_copy` returns the input in
**row-major order** instead of the unfolded strided windows — right shape, wrong elements (the
stride/gather is never applied). → [`bugs/wrongvalue_view_ops.md`](bugs/wrongvalue_view_ops.md)

### Mismatch · WRONG-VALUE (math)
`reflection_pad2d(.out)` / `reflection_pad1d.out` compute wrong reflected-border values;
`remainder.Tensor_out`/`.Scalar_out`, `pow.Scalar_out`, `var_mean.correction`, `var.correction`
diverge beyond tolerance. → [`bugs/wrongvalue_math_ops.md`](bugs/wrongvalue_math_ops.md)

### Mismatch · NONFINITE (device introduces nan)
`_native_batch_norm_legit.no_stats` emits `nan` where leaf and reference are finite (batch-1 /
zero-variance divide) — 12/12 sampled have a finite reference. →
[`bugs/nonfinite_batchnorm.md`](bugs/nonfinite_batchnorm.md)

### Mismatch · WRONG-SHAPE (`out=` variant)
`min.dim_min`, `max.dim_max`, `topk.values` return a degenerate **empty** output `(0,0,0,0,0)` — the
delegate doesn't resize the preallocated `out=` tensors (form-specific). →
[`bugs/wrongshape_out_variants.md`](bugs/wrongshape_out_variants.md)

### Crash · native abort
`max_pool2d_with_indices_backward.grad_input` — native SIGABRT (no catchable message), 55 of 58
delegated crashes; rare aborts in `unfold_copy`, `reflection_pad2d.out`, `convolution`. The missing
guard is the bug. → [`bugs/crash_maxpool_backward.md`](bugs/crash_maxpool_backward.md)

### Skip · OpenVINO delegate blob execute failure (coverage gap)
10,143 jobs / 134 ops: `CALL_DELEGATE execute failed (0x1)` — op partitions and compiles to
OpenVINO but the blob won't execute. Full per-op table in [`skips.md`](skips.md).

### Lower-confidence / not promoted
- **VALUE (fp accumulation):** `addmm.out`, `bmm.out`, `convolution`, `linear(.out)` — near-tolerance
  fp32 accumulation-order noise. → [`bugs/value_fp_accumulation.md`](bugs/value_fp_accumulation.md)
- **UB / dtype:** `bitwise_left_shift.*` (negative/large shift is UB), `prod.int_out` (dtype-form
  artifact). → [`bugs/suspect_ub_dtype.md`](bugs/suspect_ub_dtype.md)

## Per-operator table (delegated-failure ops)
Full table: [`per_op_table.tsv`](per_op_table.tsv) (203 rows). Columns: k samples, delegated vs
portable, and non-OK counts split by delegated/portable. Top delegated-failure ops:

| operator | k | mism(deleg) | crash(deleg) | skip(runtime) | mechanism |
|---|--:|--:|--:|--:|---|
| `native_group_norm` | 720 | 572 | 0 | 0 | *mostly reference-nan confound* + fp16 precision |
| `clone` | 434 | 432 | 0 | 0 | **ZEROED** |
| `lift_fresh_copy` | 432 | 426 | 0 | 4 | **ZEROED** |
| `alias_copy` | 431 | 419 | 0 | 7 | **ZEROED** |
| `reflection_pad1d.out` | 434 | 385 | 0 | 0 | ZEROED / WRONG-VALUE |
| `reflection_pad2d` | 417 | 352 | 0 | 0 | **WRONG-VALUE** (padding) |
| `t_copy` | 436 | 348 | 0 | 6 | **ZEROED** |
| `reflection_pad2d.out` | 430 | 332 | 1 | 0 | WRONG-VALUE + rare crash |
| `diagonal_copy` | 431 | 327 | 0 | 1 | **WRONG-VALUE** (row≠diag) |
| `transpose_copy.int` | 429 | 322 | 0 | 1 | **ZEROED** |
| `bitwise_left_shift.Tensor_out` | 428 | 202 | 0 | 2 | UB shift (suspect) |
| `var.correction` | 435 | 191 | 0 | 216 | WRONG-VALUE + SKIP |
| `var_mean.correction` | 436 | 181 | 0 | 230 | WRONG-VALUE + SKIP |
| `var.correction_out` | 432 | 174 | 0 | 252 | ZEROED + SKIP |
| `unfold_copy` | 435 | 133 | 1 | 0 | **WRONG-VALUE** (raw order) |
| `topk.values` | 272 | 104 | 0 | 24 | **WRONG-SHAPE** |
| `addmm.out` | 427 | 82 | 0 | 4 | fp accumulation (low sev) |
| `bmm.out` | 428 | 64 | 0 | 0 | fp accumulation (low sev) |
| `min.dim_min` | 389 | 56 | 0 | 12 | **WRONG-SHAPE** |
| `_native_batch_norm_legit.no_stats` | 428 | 49 | 0 | 0 | **NONFINITE** |
| `max.dim_max` | 386 | 36 | 0 | 9 | **WRONG-SHAPE** |
| `max_pool2d_with_indices_backward.grad_input` | 286 | 14 | 55 | 0 | **CRASH** + WRONG-VALUE |
| `linear` | 432 | 2 | 0 | 394 | fp accumulation + SKIP |

## Coverage ledger
- **82,356 jobs** produced; **203 distinct operators** produced in the corpus.
- **225 usable operator-names scheduled** (231 opnodes; the seed schedule round-robins every usable
  op equally). **203 produced → 22 scheduled but never lowered** (generation attrition, a
  generator/OpenVINO coverage gap, not a device bug): `_fft_c2r(.out)`, `_fft_r2c.out`,
  `_native_batch_norm_legit(.out/.no_stats_out)`, `cat.out`, `convolution_backward`, `fill.Tensor`,
  `masked_select(.out)`, `max_pool2d_with_indices.out`, `mean.out`, `nonzero(.out)`, `pixel_shuffle`,
  `reflection_pad3d.out`, `repeat_interleave.Tensor`, `split_copy.Tensor_out`,
  `split_with_sizes_copy.out`, `stack.out`, `unbind_copy.int_out` (data-dependent / dynamic-shape /
  multi-output forms the generator couldn't satisfy). Full list: `coverage_ops.json`.
- Every **produced** op has a verdict: OK, a confirmed bug (above), a ruled-out suspect
  ([`ruled_out/`](ruled_out/)), or a SKIP coverage gap ([`skips.md`](skips.md)).
- `covered_ops (203) + never_lowered (22) == scheduled (225)`. ✓

## Rigor notes (per the workflow)
- **Determinism-gated (3b):** 235/237 delegated failing jobs REPRO on all N=5 re-runs; only 2 (max_pool
  crash) intermittent. No single-pass verdicts.
- **Delegated-vs-portable first (3a):** 583 portable mismatches + 11 portable crashes + 866 portable
  SKIPs removed as non-OpenVINO ([`ruled_out/portable_fallback.md`](ruled_out/portable_fallback.md)).
- **Reference confound (3d/3e):** `native_group_norm` "nonfinite" is PyTorch-side nan, not device
  ([`ruled_out/native_group_norm_reference_nan.md`](ruled_out/native_group_norm_reference_nan.md)).
- **CRASH reasons recorded (4):** all native aborts, no catchable message; the missing guard is the bug.
- **SKIP reasons recorded verbatim (4/6):** [`skips.md`](skips.md) splits OpenVINO blob-execute
  failures (0x1) from portable guards (0x12).

## Reproduce
Bring up an **isolated** pipeline (the user runs their own brokers on 15554/15574 — use different
ports):
```
# broker
PYTHONPATH=/data/jwen929 .venv/bin/python -m mobile broker --job-port 15664 --client-port 15665 --ctrl-port 15666
# worker(s) — MOBILE_BACKENDS=openvino avoids the QNN/OpenVINO in-process heap conflict;
# --job-timeout 120 prevents warm-child respawn spirals under load
MOBILE_BACKENDS=openvino PYTHONPATH=/data/jwen929 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python openvino_client/openvino_client.py --client-port 15665 --ctrl-port 15666 --job-timeout 120
# feed + verify
PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/bugs/repro.py
```

## Files
- `README.md` — this synthesis.
- `bugs/` — confirmed bugs by mechanism + `INDEX.md` + `repro.py` (device repro).
- `skips.md` — SKIP reason tables (OV blob-execute vs portable guards) + CRASH tables.
- `ruled_out/` — portable-fallback + reference-nan suspects with the filter that removed them.
- `per_op_table.tsv` — per-operator coverage ledger (203 ops). `skiplog.tsv` — raw non-OK rows.
- `manifest.tsv` — per-job op+delegation. `buckets.json` / `determinism.json` / `coverage*.json`.
- Scripts: `build_manifest.py`, `analyze.py`, `determinism_gate.py`, `inspect_jobs.py`,
  `build_skips.py`, `replay.py`.
