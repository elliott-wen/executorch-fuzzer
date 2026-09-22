# Core ML backend (Apple-Silicon Mac) — single-operator fuzzing findings

ExecuTorch Core ML delegate, exercised with a **single-operator** corpus (`corpus_v3/coreml`,
generated with `--nodes 1`: each graph is exactly one aten op wired to leaf inputs). Every non-OK
outcome is therefore already a minimal repro attributed to one operator. Fed to a real
**Apple-Silicon Mac** Core ML worker over the ZeroMQ broker; the feeder holds the eager CPU
reference and diffs each device output ([executor/compare.py](../../executor/compare.py)).

**Core rule followed: every claim below is device-verified.** Suspects were re-run on the device and
cleared through the step-3 confound filters before being called a bug.

## Run summary (52,970 graphs, one pass)

| outcome | bulk count | after verification |
|---------|-----------:|--------------------|
| OK       | 47,556 | — |
| MISMATCH | 1,467  | 28 delegated ops consistent (5/5); confounds removed |
| CRASH    | 268    | **~48% were collateral**; 14 ops reproduce (native abort) |
| TIMEOUT  | 26     | 3 ops reproduce (hang) |
| SKIP     | 3,653  | coverage gaps — [skips.md](skips.md) |

The headline methodology result: at the bulk `--window 64`, a native abort takes down its in-flight
neighbours, which are then mislogged as `native abort`. Re-running every crash candidate at
`--window 1` (one abort = one lost job) plus an N=5 same-job gate cut the "crash" set roughly in half.
The delegated MISMATCH set, by contrast, reproduced **5/5 with zero flakiness** (these are float,
deterministic — not int8 near-tolerance noise).

## Confirmed bugs
Full index with mechanisms and repros: [bugs/INDEX.md](bugs/INDEX.md). Summary:

### Wrong value (MISMATCH, delegated, finite inputs, 5/5)
- **`atan2.out`** — `atan2(0, x<0)` returns `0` instead of `π` (missing quadrant correction). [bugs/atan2.md](bugs/atan2.md)
- **`remainder.Scalar`** — output all-**zeroed**. [bugs/remainder-scalar.md](bugs/remainder-scalar.md)
- **`mean`** (int input) — **integer-truncated** mean instead of float mean. [bugs/mean-int.md](bugs/mean-int.md)
- **`_upsample_bilinear2d_aa.out`** — antialiasing ignored (Δ up to 2.33). [bugs/upsample-bilinear2d-aa.md](bugs/upsample-bilinear2d-aa.md)
- **`avg_pool2d.out`** — wrong pooling divisor (~2×). [bugs/avg_pool2d.md](bugs/avg_pool2d.md)
- **`bitwise_left_shift.Tensor_out`** — int shift overflow → `-2³¹` / `-inf`. [bugs/bitwise-left-shift.md](bugs/bitwise-left-shift.md)
- **`max_pool2d_with_indices_backward.grad_input`** — wrong gradient scatter (also crashes). [bugs/max_pool2d-backward.md](bugs/max_pool2d-backward.md)
- **precision** (`var.correction`, `var.correction_out` fp16, `_native_batch_norm_legit.no_stats`, `logit`, `pow.Scalar_out`, `remainder.Tensor_out`, `convolution`) — consistent, beyond tolerance. [bugs/precision-reductions.md](bugs/precision-reductions.md)

### Native-abort crashes (no catchable message) — [bugs/crash-native-abort.md](bugs/crash-native-abort.md)
`unfold_copy`, `prod.int_out`, `diagonal_copy`, `scatter.value_out`, `max_pool2d_with_indices_backward.grad_input`,
`linear.out`, `mean.out`, `cumsum.out`, `logical_or.out`, `mm.out`, `t_copy`, `tril.out`, `alias_copy`, `view_copy`.

### Hangs (reproducible TIMEOUT) — [bugs/hang-timeout.md](bugs/hang-timeout.md)
`slice_scatter`, `select_scatter`, `max.unary_out`.

### Coverage gaps (SKIP) — [skips.md](skips.md)
3,653 runtime rejections across 65 ops. Dominant reasons: **"Failed to compile model"** (1,377 — the
Core ML `mlassetio` model-spec parse failure, e.g. `_softmax`, `_log_softmax`, `index_put`, `clamp`,
`amin`/`amax`, `sum.IntList`), **"Caught an unknown exception"** (1,075 — `any.*`, `var.correction_out`,
`mean.out`, `min/max.unary`), **"Attempted to resize a static tensor"** (316 — `avg_pool2d`,
`min.dim`, `max.dim`), and dtype/rank gaps (`_fft_r2c` onesided/dim-order, `_pdist_forward` rank +
`Unhandled dtype Long/Bool`, `DataType not supported` for several bitwise ops).

## Ruled out (NOT Core ML bugs) — [ruled_out/README.md](ruled_out/README.md)
- **Portable-fallback** mismatches (`delegated.ops=0`, the delegate never ran the op): `bitwise_left_shift.Tensor_Scalar` (199), `grid_sampler_2d` (81), `prod`/`prod.out` (111), `fmod.Scalar` (18), … — portable/reference issues.
- **Empty-tensor reductions** returning the dtype sentinel (reference degenerate): `min.unary_out`, `max.unary_out`, `mean.out` mismatch forms.
- **Non-finite INPUT propagation** (leaf already `inf`/`nan`): `gelu.out`, `floor_divide`, `prod.int_out` mismatch form, `pow.Tensor_Tensor_out`, `_native_batch_norm_legit_no_training`.
- **Collateral crashes** (re-ran OK at window=1): `mul.out` + ~40 ops.

## Coverage ledger
- **Operators produced in corpus:** 196 distinct (of ~228 scheduled usable ops → ~32 never lowered / seed-infeasible for Core ML — a generator/backend coverage gap, not a device bug).
- **Samples per op:** median 287, range 3–864 — every produced op is far above the k≥5 determinism floor.
- **Verdicts:** 79 ops fully OK across all samples; 117 ops have ≥1 non-OK (table below). Every non-OK op is either a confirmed bug (bugs/), a SKIP coverage gap ([skips.md](skips.md)), or a ruled-out confound ([ruled_out/](ruled_out/README.md)).

## What this corpus cannot catch
Single-op graphs have no siblings, so multi-op faults never trigger: memory planning, buffer
aliasing / copy-elision, cross-op fusion, dtype/layout rewriting across nodes. Every finding here is
attributable to one operator (including its own lowering).

## Per-operator table (operators with ≥1 non-OK; 79 fully-OK ops omitted)
Columns: total samples, OK, MISMATCH, CRASH, TIMEOUT, SKIP, and the delegated-vs-portable split of the
non-OK rows (`deleg`=Core ML kernel ran; `port`=portable fallback).

| operator | samples | OK | MIS | CRASH | TMO | SKIP | delegated-vs-portable (non-OK) |
|----------|--------:|---:|----:|------:|----:|-----:|--------------------------------|
| `bitwise_left_shift.Tensor_Scalar` | 433 | 234 | 199 | 0 | 0 | 0 | MISMATCH:port=199 |
| `var.correction` | 222 | 59 | 158 | 0 | 0 | 5 | MISMATCH:deleg=158 SKIP:deleg=5 |
| `bitwise_left_shift.Tensor_out` | 408 | 261 | 143 | 0 | 0 | 4 | MISMATCH:deleg=20 MISMATCH:port=123 SKIP:deleg=4 |
| `grid_sampler_2d` | 265 | 184 | 81 | 0 | 0 | 0 | MISMATCH:port=81 |
| `gelu.out` | 103 | 24 | 77 | 2 | 0 | 0 | CRASH:deleg=2 MISMATCH:deleg=77 |
| `var.correction_out` | 359 | 6 | 76 | 1 | 0 | 276 | CRASH:deleg=1 MISMATCH:deleg=76 SKIP:deleg=276 |
| `_upsample_bilinear2d_aa.out` | 434 | 357 | 77 | 0 | 0 | 0 | MISMATCH:deleg=77 |
| `mean` | 344 | 261 | 69 | 3 | 0 | 11 | CRASH:deleg=3 MISMATCH:deleg=69 SKIP:deleg=11 |
| `bitwise_left_shift.Tensor_Scalar_out` | 438 | 365 | 72 | 0 | 0 | 1 | MISMATCH:port=72 SKIP:deleg=1 |
| `prod.out` | 434 | 373 | 61 | 0 | 0 | 0 | MISMATCH:port=61 |
| `_native_batch_norm_legit.no_stats` | 243 | 188 | 55 | 0 | 0 | 0 | MISMATCH:deleg=55 |
| `atan2.out` | 236 | 183 | 53 | 0 | 0 | 0 | MISMATCH:deleg=53 |
| `floor_divide.out` | 287 | 237 | 50 | 0 | 0 | 0 | MISMATCH:deleg=50 |
| `prod` | 439 | 389 | 50 | 0 | 0 | 0 | MISMATCH:port=50 |
| `min.unary_out` | 213 | 115 | 34 | 1 | 0 | 63 | CRASH:deleg=1 MISMATCH:deleg=34 SKIP:deleg=63 |
| `floor_divide` | 255 | 223 | 32 | 0 | 0 | 0 | MISMATCH:deleg=32 |
| `pow.Scalar_out` | 337 | 306 | 28 | 3 | 0 | 0 | CRASH:deleg=3 MISMATCH:deleg=28 |
| `max.unary_out` | 189 | 101 | 29 | 0 | 1 | 58 | MISMATCH:deleg=29 SKIP:deleg=58 TIMEOUT:deleg=1 |
| `mul.out` | 321 | 292 | 0 | 29 | 0 | 0 | CRASH:deleg=29 |
| `prod.int_out` | 417 | 366 | 10 | 17 | 0 | 24 | CRASH:deleg=17 MISMATCH:deleg=10 SKIP:deleg=24 |
| `unfold_copy` | 281 | 256 | 0 | 25 | 0 | 0 | CRASH:deleg=25 |
| `max_pool2d_with_indices_backward.grad_input` | 278 | 253 | 11 | 14 | 0 | 0 | CRASH:deleg=14 MISMATCH:deleg=11 |
| `view_copy` | 321 | 298 | 0 | 23 | 0 | 0 | CRASH:deleg=23 |
| `avg_pool2d.out` | 284 | 45 | 10 | 12 | 0 | 217 | CRASH:deleg=12 MISMATCH:deleg=10 SKIP:deleg=217 |
| `bitwise_right_shift.Tensor_Scalar` | 433 | 413 | 20 | 0 | 0 | 0 | MISMATCH:port=20 |
| `logit` | 336 | 315 | 19 | 0 | 0 | 2 | MISMATCH:deleg=19 SKIP:deleg=2 |
| `fmod.Scalar` | 415 | 397 | 18 | 0 | 0 | 0 | MISMATCH:port=18 |
| `slice_scatter` | 284 | 267 | 0 | 1 | 16 | 0 | CRASH:deleg=1 TIMEOUT:deleg=16 |
| `diagonal_copy` | 284 | 267 | 0 | 17 | 0 | 0 | CRASH:deleg=17 |
| `scatter.value_out` | 16 | 0 | 0 | 16 | 0 | 0 | CRASH:deleg=16 |
| `abs` | 281 | 266 | 0 | 13 | 0 | 2 | CRASH:deleg=13 SKIP:deleg=2 |
| `sum.IntList_out` | 351 | 228 | 9 | 0 | 0 | 114 | MISMATCH:deleg=9 SKIP:deleg=114 |
| `select_scatter` | 282 | 273 | 0 | 0 | 9 | 0 | TIMEOUT:deleg=9 |
| `bmm.out` | 341 | 333 | 0 | 7 | 0 | 1 | CRASH:deleg=7 SKIP:deleg=1 |
| `any.all_out` | 279 | 167 | 6 | 0 | 0 | 106 | MISMATCH:deleg=6 SKIP:deleg=106 |
| `logical_or.out` | 340 | 334 | 0 | 6 | 0 | 0 | CRASH:deleg=6 |
| `linear.out` | 26 | 15 | 0 | 5 | 0 | 6 | CRASH:deleg=5 SKIP:deleg=6 |
| `any.dims_out` | 706 | 153 | 4 | 0 | 0 | 549 | MISMATCH:deleg=4 SKIP:deleg=549 |
| `mean.out` | 339 | 99 | 1 | 3 | 0 | 236 | CRASH:deleg=3 MISMATCH:deleg=1 SKIP:deleg=236 |
| `max.dim_max` | 376 | 326 | 0 | 4 | 0 | 46 | CRASH:deleg=4 SKIP:deleg=46 |
| `cumsum.out` | 272 | 266 | 0 | 4 | 0 | 2 | CRASH:deleg=4 SKIP:deleg=2 |
| `eq.Tensor_out` | 345 | 340 | 0 | 4 | 0 | 1 | CRASH:deleg=4 SKIP:deleg=1 |
| `narrow_copy` | 864 | 859 | 0 | 4 | 0 | 1 | CRASH:port=4 SKIP:port=1 |
| `sign.out` | 173 | 169 | 0 | 4 | 0 | 0 | CRASH:deleg=4 |
| `minimum.out` | 206 | 202 | 0 | 4 | 0 | 0 | CRASH:deleg=4 |
| `split_copy.Tensor` | 148 | 123 | 0 | 3 | 0 | 22 | CRASH:deleg=3 SKIP:deleg=22 |
| `convolution` | 67 | 50 | 3 | 0 | 0 | 14 | MISMATCH:deleg=3 SKIP:deleg=14 |
| `remainder.Scalar` | 238 | 234 | 3 | 0 | 0 | 1 | MISMATCH:deleg=3 SKIP:deleg=1 |
| `native_layer_norm` | 302 | 299 | 0 | 3 | 0 | 0 | CRASH:deleg=3 |
| `_pdist_forward` | 436 | 82 | 2 | 0 | 0 | 352 | MISMATCH:port=2 SKIP:port=352 |
| `split_with_sizes_copy` | 336 | 216 | 0 | 2 | 0 | 118 | CRASH:deleg=2 SKIP:deleg=118 |
| `min.dim_min` | 385 | 325 | 0 | 2 | 0 | 58 | CRASH:deleg=2 SKIP:deleg=58 |
| `logical_not` | 321 | 318 | 0 | 2 | 0 | 1 | CRASH:deleg=2 SKIP:deleg=1 |
| `ne.Scalar_out` | 346 | 343 | 0 | 2 | 0 | 1 | CRASH:deleg=2 SKIP:deleg=1 |
| `logit.out` | 327 | 325 | 0 | 2 | 0 | 0 | CRASH:deleg=2 |
| `copy` | 335 | 333 | 0 | 2 | 0 | 0 | CRASH:deleg=2 |
| `pow.Tensor_Tensor_out` | 302 | 300 | 2 | 0 | 0 | 0 | MISMATCH:deleg=2 |
| `logical_xor.out` | 342 | 340 | 1 | 1 | 0 | 0 | CRASH:deleg=1 MISMATCH:deleg=1 |
| `_log_softmax.out` | 269 | 139 | 0 | 1 | 0 | 129 | CRASH:deleg=1 SKIP:deleg=129 |
| `mul.Scalar` | 326 | 321 | 0 | 1 | 0 | 4 | CRASH:deleg=1 SKIP:deleg=4 |
| `unsqueeze_copy` | 268 | 264 | 0 | 1 | 0 | 3 | CRASH:deleg=1 SKIP:deleg=3 |
| `masked_fill.Scalar` | 42 | 39 | 0 | 1 | 0 | 2 | CRASH:deleg=1 SKIP:deleg=2 |
| `erf.out` | 180 | 178 | 0 | 1 | 0 | 1 | CRASH:deleg=1 SKIP:deleg=1 |
| `bitwise_right_shift.Tensor_Scalar_out` | 429 | 427 | 1 | 0 | 0 | 1 | MISMATCH:port=1 SKIP:deleg=1 |
| `relu` | 236 | 234 | 0 | 1 | 0 | 1 | CRASH:deleg=1 SKIP:deleg=1 |
| `where.self_out` | 301 | 299 | 0 | 1 | 0 | 1 | CRASH:deleg=1 SKIP:deleg=1 |
| `logical_not.out` | 330 | 329 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `alias_copy` | 289 | 288 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `squeeze_copy.dims` | 298 | 297 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `bitwise_xor.Scalar_out` | 365 | 364 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `t_copy` | 306 | 305 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `bitwise_not.out` | 433 | 432 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `where.self` | 154 | 153 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `clone` | 281 | 280 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `tril.out` | 348 | 347 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `logical_and.out` | 351 | 350 | 1 | 0 | 0 | 0 | MISMATCH:deleg=1 |
| `squeeze_copy.dim` | 294 | 293 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `reflection_pad2d` | 109 | 108 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `_native_batch_norm_legit_no_training` | 331 | 330 | 1 | 0 | 0 | 0 | MISMATCH:deleg=1 |
| `sigmoid.out` | 337 | 336 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `addmm.out` | 91 | 90 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `tan.out` | 210 | 209 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `gt.Tensor_out` | 275 | 274 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `acos.out` | 187 | 186 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `unbind_copy.int` | 357 | 356 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `remainder.Tensor_out` | 246 | 245 | 1 | 0 | 0 | 0 | MISMATCH:deleg=1 |
| `select_copy.int` | 322 | 321 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `mm.out` | 322 | 321 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `expm1.out` | 190 | 189 | 0 | 1 | 0 | 0 | CRASH:deleg=1 |
| `index_put` | 429 | 0 | 0 | 0 | 0 | 429 | SKIP:deleg=429 |
| `_fft_r2c` | 435 | 67 | 0 | 0 | 0 | 368 | SKIP:port=368 |
| `_softmax.out` | 254 | 119 | 0 | 0 | 0 | 135 | SKIP:deleg=135 |
| `mean.dtype_out` | 221 | 140 | 0 | 0 | 0 | 81 | SKIP:deleg=81 |
| `clamp.Tensor_out` | 67 | 2 | 0 | 0 | 0 | 65 | SKIP:deleg=65 |
| `pixel_shuffle` | 49 | 18 | 0 | 0 | 0 | 31 | SKIP:deleg=31 |
| `index_select.out` | 57 | 35 | 0 | 0 | 0 | 22 | SKIP:deleg=22 |
| `index_select` | 59 | 41 | 0 | 0 | 0 | 18 | SKIP:deleg=18 |
| `amin.out` | 59 | 43 | 0 | 0 | 0 | 16 | SKIP:deleg=16 |
| `amax.out` | 54 | 39 | 0 | 0 | 0 | 15 | SKIP:deleg=15 |
| `replication_pad2d.out` | 23 | 17 | 0 | 0 | 0 | 6 | SKIP:deleg=6 |
| `max` | 184 | 179 | 0 | 0 | 0 | 5 | SKIP:deleg=5 |
| `bitwise_xor.Tensor_out` | 420 | 416 | 0 | 0 | 0 | 4 | SKIP:deleg=4 |
| `bitwise_right_shift.Tensor_out` | 391 | 387 | 0 | 0 | 0 | 4 | SKIP:deleg=4 |
| `bitwise_or.Tensor_out` | 419 | 416 | 0 | 0 | 0 | 3 | SKIP:deleg=3 |
| `embedding` | 3 | 1 | 0 | 0 | 0 | 2 | SKIP:deleg=2 |
| `div.out_mode` | 298 | 296 | 0 | 0 | 0 | 2 | SKIP:deleg=2 |
| `flip` | 163 | 161 | 0 | 0 | 0 | 2 | SKIP:deleg=2 |
| `div.Scalar` | 325 | 323 | 0 | 0 | 0 | 2 | SKIP:deleg=2 |
| `glu.out` | 135 | 134 | 0 | 0 | 0 | 1 | SKIP:deleg=1 |
| `round.out` | 86 | 85 | 0 | 0 | 0 | 1 | SKIP:deleg=1 |
| `isnan` | 282 | 281 | 0 | 0 | 0 | 1 | SKIP:deleg=1 |
| `topk.values` | 8 | 7 | 0 | 0 | 0 | 1 | SKIP:deleg=1 |
| `log2.out` | 287 | 286 | 0 | 0 | 0 | 1 | SKIP:deleg=1 |
| `le.Tensor_out` | 259 | 258 | 0 | 0 | 0 | 1 | SKIP:deleg=1 |
| `neg.out` | 345 | 344 | 0 | 0 | 0 | 1 | SKIP:deleg=1 |
| `ge.Tensor_out` | 221 | 220 | 0 | 0 | 0 | 1 | SKIP:deleg=1 |
| `transpose_copy.int` | 343 | 342 | 0 | 0 | 0 | 1 | SKIP:deleg=1 |

## Reproduce
Broker + a Mac Core ML worker must be up (a human sets this up):
```
python -m mobile broker --job-port 15554 --client-port 15555 --ctrl-port 15556
# Mac worker connects on client-port 15555
```
Then any repro, e.g.:
```
PYTHONPATH=/data/jwen929 .venv/bin/python findings/coreml_mac/bugs/repro_atan2.py
```
Each `bugs/repro_*.py` streams its exact corpus jobs through the broker and prints eager-vs-device.
