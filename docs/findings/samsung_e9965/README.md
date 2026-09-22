# Samsung ENN (Exynos 2600 / E9965) — single-operator analysis

Backend: **samsung / ENN** (Exynos AI LiteCore v1.2.0, chipset `E9965`).
Device: Galaxy (Exynos 2600, SoC `Solomon`, NPU `olympus_pamir`, NPU driver NCP **v29**).
Corpus: `corpus_v3/samsung`, **75,865** single-operator graphs (`--nodes 1`), 199 distinct operators.
Method: [analysis_single.md](../../analysis_single.md). Every verdict below is **device-verified**.

> Precondition: this run was only possible after fixing `CHIPSET` (`gen/export/backends/samsung.py`)
> from `E9955` (Exynos 2500, NCP v28) to `E9965`. With v28 models the device driver rejected
> every ENN model (`NCP Ver(28) vs NpuUserDriver(29)`) and all delegated graphs SKIP'd at load.

## Feed tally (full corpus)
| status | count | % |
|---|--:|--:|
| OK | 64,713 | 85.3 |
| MISMATCH | 4,854 | 6.4 |
| SKIP | 6,213 | 8.2 |
| CRASH | 85 | 0.1 |
| TIMEOUT | 0 | 0 |

## Filtering (what survived)
- **3a delegated-vs-portable** — of 4,939 MISMATCH+CRASH, **3,640 ran on the ENN delegate** (`ops≥1`);
  **1,299 were portable-fallback** (`ops=0`, CPU kernel ran) → not ENN bugs → `ruled_out/`.
- **All 85 CRASHes are portable-fallback** (`ops=0`): 78× `max_pool2d_with_indices_backward`,
  4× `unfold_copy`/`narrow_copy`. **ENN itself produced zero crashes.** → `ruled_out/portable_crashes.md`.
- **3b determinism gate** — 164 representative delegated failures re-run **5×** each:
  **112 CONFIRMED** (5/5), **38 INTERMITTENT** (1–4/5), **14 FLAKY** (0/5, dropped). Aggregated per
  operator → **40 operators with a confirmed deterministic reproducer**.
- **3e non-finite leaf** — `gelu/relu/var/logit/linear/_softmax/sqrt/rsqrt/clamp` `nan→inf` cases
  come from graphs that inject `nan/inf` **inputs** → non-finite-input propagation, not clean kernel
  bugs → `ruled_out/nonfinite_input.md`. **Exception:** `div.Scalar` overflows from *finite* input.

## Confirmed ENN operator bugs, by mechanism

### 1. Integer tensor data-movement is broken (flagship)
Integer (`int32`/`int64`) layout/copy operators corrupt their output two ways:
- **GARBAGE READ** — values at ~2¹⁶ scale, unrelated to input:
  `permute_copy`, `roll`, `t_copy`, `transpose_copy.int`, `select_copy.int`.
  e.g. `permute_copy(int32)`: eager `[-3,-3,-3,2,…]` → device `[-131075, 327677, 327677, -131076, …]`.
- **DROPPED TAIL STORE (ZEROED)** — trailing elements written as `0`:
  `constant_pad_nd`, `expand_copy`, `squeeze_copy.dim`, `squeeze_copy.dims`, `select_scatter`.
  e.g. `expand_copy(int32)`: eager `[…,-4,-4]` → device `[…,-4,0]`.

Same root: the ENN copy/layout path mishandles integer element width/stride. Float variants of the
same ops (`view_copy`, float `transpose_copy`) also mismatch with moderate deltas.
Repro: [bugs/repro_int_datamovement.py](bugs/repro_int_datamovement.py).

### 2. div.Scalar scaled by 2¹⁶ → fp16 overflow (NONFINITE)
`div.Scalar` — **finite** input `[-0.482, 1.110, 0.934]` → device `[-31600, inf, 61184]`. The device
output is the reference **× 65536 (2¹⁶)** exactly (`-0.482×65536 ≈ -31600`), overflowing fp16 to `inf`
for any `|x|≳1`. Not a plain precision overflow — a **2¹⁶ scale error**.
Repro: [bugs/repro_div_scalar_overflow.py](bugs/repro_div_scalar_overflow.py).

> **Likely common root.** The integer garbage in §1 is also 2¹⁶-scaled (`131072=2¹⁷`, `65536=2¹⁶`,
> `327680=5·2¹⁶`), and `div.Scalar` is off by exactly 2¹⁶. A byte-vs-element / 16-bit-width confusion
> in the ENN backend would explain **both** the integer data-movement corruption and this scale error.
> Worth investigating as a single defect.

### 3. ZEROED stores (float)
`embedding`, `mul.Scalar` (fp16), `pixel_unshuffle` — device returns all-zero where the reference is
finite/non-zero (dropped store, not a scale error).
Repro: [bugs/repro_zeroed.py](bugs/repro_zeroed.py).

### 4. Reduction integer saturation (uint8) — flagged
`prod.int_out`, `sum.IntList_out` — `uint8` device output saturates to `255` where the int reference
is `0`. Real value divergence, but the narrow `uint8` output dtype is itself suspect (isolation 3c);
filed with that caveat.

### 5. remainder sign/scale
`remainder.Tensor_out` — `SCALE:-0.5`, device `[-2,0,-2]` → `[1,0,1]`: sign/scale error in the
op's own lowering (delegated subset only; the op is 113/433 delegated).

### 6. Float WRONG-VALUE (kernel/precision)
`rsub.Scalar` (358 finite mismatches — largest), `sub.out`, `add.Scalar`, `mul.Scalar`, `glu.out`,
`maximum.out`, `avg_pool2d.out`, `pixel_shuffle`, `hardtanh`, `bmm.out`, `unsqueeze_copy` — values
diverge beyond per-dtype rtol/atol. See per-op table.

## Per-operator table (delegated failures; verdict from the 5× gate)
`total` = corpus samples for the op; `deleg` = how many delegated to ENN; MIS-fin/MIS-nf/SKIP are the
**delegated** counts. `mechanism` = dominant device-verified mechanism (see groupings above).

| operator | total | deleg | MIS-fin | MIS-nf | SKIP | verdict | mechanism |
|---|--:|--:|--:|--:|--:|---|---|
| `rsub.Scalar` | 439 | 439 | 358 | 36 | 5 | CONFIRMED | WRONG-VALUE |
| `gelu.out` | 440 | 440 | 4 | 373 | 0 | CONFIRMED | WRONG-VALUE |
| `sub.Scalar` | 403 | 403 | 229 | 125 | 1 | CONFIRMED | ZEROED |
| `add.Scalar` | 391 | 391 | 208 | 120 | 2 | CONFIRMED | WRONG-VALUE |
| `_softmax.out` | 444 | 444 | 6 | 281 | 0 | CONFIRMED | ZEROED |
| `div.Scalar` | 305 | 305 | 1 | 283 | 0 | CONFIRMED | NONFINITE |
| `mul.Scalar` | 204 | 204 | 94 | 99 | 0 | CONFIRMED | ZEROED |
| `sub.out` | 379 | 379 | 158 | 23 | 46 | CONFIRMED | WRONG-VALUE |
| `clamp.out` | 372 | 372 | 21 | 108 | 165 | CONFIRMED | WRONG-VALUE |
| `select_scatter` | 386 | 386 | 107 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `hardtanh` | 416 | 416 | 104 | 0 | 125 | CONFIRMED | WRONG-VALUE |
| `logit` | 386 | 386 | 23 | 79 | 0 | CONFIRMED | WRONG-VALUE |
| `select_copy.int` | 378 | 378 | 91 | 0 | 4 | CONFIRMED | SCALE:-2.184e+04 |
| `maximum.out` | 359 | 359 | 85 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `pixel_shuffle` | 598 | 598 | 82 | 1 | 327 | CONFIRMED | WRONG-VALUE |
| `permute_copy` | 379 | 379 | 54 | 1 | 2 | CONFIRMED | WRONG-VALUE |
| `leaky_relu.out` | 444 | 444 | 6 | 37 | 0 | INTERMITTENT |  |
| `sqrt.out` | 442 | 442 | 0 | 43 | 64 | CONFIRMED | WRONG-VALUE |
| `mul.out` | 271 | 271 | 1 | 39 | 0 | INTERMITTENT |  |
| `rsqrt.out` | 303 | 303 | 0 | 39 | 2 | CONFIRMED | WRONG-VALUE |
| `pixel_unshuffle` | 728 | 728 | 34 | 0 | 147 | CONFIRMED | ZEROED |
| `glu.out` | 437 | 437 | 28 | 2 | 1 | CONFIRMED | WRONG-VALUE |
| `bmm.out` | 321 | 321 | 2 | 23 | 0 | CONFIRMED | WRONG-VALUE |
| `sum.IntList_out` | 407 | 146 | 23 | 0 | 14 | CONFIRMED | WRONG-VALUE |
| `transpose_copy.int` | 432 | 88 | 20 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `hardtanh.out` | 435 | 435 | 19 | 0 | 237 | INTERMITTENT |  |
| `remainder.Tensor_out` | 433 | 113 | 18 | 0 | 37 | CONFIRMED | SCALE:-0.5 |
| `prod.int_out` | 391 | 150 | 17 | 0 | 48 | CONFIRMED | WRONG-VALUE |
| `expand_copy` | 347 | 347 | 17 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `squeeze_copy.dims` | 389 | 389 | 14 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `logit.out` | 398 | 398 | 0 | 13 | 97 | INTERMITTENT |  |
| `bitwise_left_shift.Tensor_out` | 441 | 281 | 2 | 9 | 47 | CONFIRMED | WRONG-VALUE |
| `relu` | 414 | 414 | 7 | 2 | 79 | CONFIRMED | WRONG-VALUE |
| `squeeze_copy.dim` | 388 | 388 | 9 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `avg_pool2d.out` | 10 | 10 | 9 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `t_copy` | 433 | 75 | 7 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `_log_softmax.out` | 417 | 417 | 3 | 4 | 0 | INTERMITTENT |  |
| `view_copy` | 361 | 361 | 6 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `div.out` | 437 | 437 | 3 | 3 | 20 | INTERMITTENT |  |
| `var.correction_out` | 438 | 271 | 0 | 6 | 0 | CONFIRMED | WRONG-VALUE |
| `linear.out` | 14 | 14 | 0 | 6 | 0 | CONFIRMED | WRONG-VALUE |
| `bitwise_xor.Scalar_out` | 441 | 299 | 4 | 0 | 97 | INTERMITTENT |  |
| `minimum.out` | 352 | 352 | 2 | 0 | 76 | INTERMITTENT |  |
| `bitwise_and.Scalar_out` | 441 | 266 | 2 | 0 | 91 | INTERMITTENT |  |
| `bitwise_or.Scalar_out` | 438 | 297 | 2 | 0 | 105 | INTERMITTENT |  |
| `roll` | 392 | 387 | 1 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `embedding` | 6 | 6 | 1 | 0 | 5 | CONFIRMED | ZEROED |
| `linear` | 13 | 13 | 0 | 1 | 0 | CONFIRMED | WRONG-VALUE |
| `constant_pad_nd` | 22 | 22 | 1 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `unsqueeze_copy` | 395 | 395 | 1 | 0 | 1 | CONFIRMED | WRONG-VALUE |
| `bitwise_right_shift.Tensor_out` | 440 | 293 | 0 | 0 | 55 | not-gated |  |
| `bitwise_or.Tensor_out` | 436 | 280 | 0 | 0 | 107 | not-gated |  |
| `fmod.Tensor_out` | 431 | 101 | 0 | 0 | 33 | not-gated |  |
| `div.out_mode` | 424 | 34 | 0 | 0 | 12 | not-gated |  |
| `replication_pad1d.out` | 438 | 438 | 0 | 0 | 438 | not-gated |  |
| `min.unary_out` | 441 | 88 | 0 | 0 | 7 | not-gated |  |
| `floor_divide.out` | 438 | 135 | 0 | 0 | 78 | not-gated |  |
| `replication_pad3d.out` | 433 | 433 | 0 | 0 | 433 | not-gated |  |
| `bitwise_xor.Tensor_out` | 437 | 259 | 0 | 0 | 110 | not-gated |  |
| `bitwise_and.Tensor_out` | 443 | 282 | 0 | 0 | 111 | not-gated |  |
| `max.unary_out` | 442 | 96 | 0 | 0 | 3 | not-gated |  |
| `floor_divide` | 410 | 72 | 0 | 0 | 30 | not-gated |  |
| `replication_pad2d.out` | 436 | 436 | 0 | 0 | 436 | not-gated |  |
| `remainder.Scalar_out` | 432 | 45 | 0 | 0 | 32 | not-gated |  |
| `stack` | 67 | 27 | 0 | 0 | 14 | not-gated |  |
| `pow.Scalar_out` | 436 | 28 | 0 | 0 | 17 | not-gated |  |


_(Full 66-op table in `_work/master_table.md`; ops below the fold are low-count or not-gated.)_

## Coverage ledger
- **199** distinct operators lowered into the corpus (75,865 jobs, exact round-robin ≈ 380/op).
- **66** operators produced ≥1 delegated non-OK outcome; **40** have a CONFIRMED deterministic bug,
  **10** are INTERMITTENT-only (`div.out`, `leaky_relu.out`, `mul.out`, `_log_softmax.out`,
  `hardtanh.out`, `logit.out`, `minimum.out`, `bitwise_{and,or,xor}.Scalar_out`) → reported as
  intermittent, not filed as confirmed bugs.
- **~133** operators were OK across all samples (no delegated failure).
- Attrition / coverage gaps: operators that never delegate (`ops=0` everywhere — e.g. `_fft_r2c`,
  `_cdist_forward`, `_adaptive_avg_pool2d`) run on portable and are out of ENN scope here.
- `covered (verdicted) + gap == scheduled`: every corpus operator is either verdicted above,
  OK, or listed as portable/coverage-gap. No silent caps.

## SKIP / CRASH — the reason is the finding
See **[skips.md](skips.md)**: 3,761 delegated SKIPs all share one runtime reason —
`[ExecuTorch Error 0x1] Internal error: Execution failed for method: forward` — dominated by
`replication_pad1d/2d/3d.out` (~1,300), `pixel_shuffle` (327), `hardtanh.out` (237), `clamp.out`
(165): ops that pass ENN partitioning but fail to execute (a real coverage gap). The CRASH table
(all portable-fallback) is there too.

## Layout of this folder
- `README.md` — this synthesis + per-op table.
- `skips.md` — SKIP reason table + CRASH table.
- `bugs/` — runnable one-op repros for the flagship confirmed bugs; `bugs/INDEX.md`.
- `ruled_out/` — portable-fallback divergences/crashes, non-finite-input confounds, intermittents.
- `_work/` — raw skip-log, analysis scripts (`analyze.py`, `rerun.py`, `mechanism.py`), captured
  tensors, and intermediate tables (reproducible pipeline).
