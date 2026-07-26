# cortex-m confirmed operator bugs

Single-operator corpus (`corpus_v3/cortex-m`, `--nodes 1`, int8/`quant=ALWAYS`), run on the
Corstone-300 FVP. Each entry is device-verified, determinism-gated (N≥5), and attributed to a
`cortex_m::` kernel by re-lowering (filter 3a). Method: `analysis_single.md`.

## MISMATCH — WRONG-VALUE
| operator | cortex_m kernel | mechanism | eager → device | determinism | repro |
|----------|-----------------|-----------|----------------|-------------|-------|
| [`permute_copy`](permute-copy-transpose.md) | `cortex_m::transpose` | general/identity permute lowered to a fixed axis-swap → output is the transpose, not the permutation | `[…,1.164,…]` → `[…,input^T…]` (max\|Δ\|=2.075) | REPRO 5/5 | [repro_permute-copy-transpose.py](repro_permute-copy-transpose.py) |

**1 confirmed compute-kernel value bug.** It is the *only* CM-COMPUTE MISMATCH in the whole corpus
(8/8 `permute_copy` mismatches; every other MISMATCH is a portable-kernel divergence — see
[../ruled_out/](../ruled_out/)).

## Kernel-guard / lowering SKIPs (reasons in [../skips.md](../skips.md))
Not given their own repro files (the runtime reason **is** the finding), but cortex-m-attributed:
- `cortex_m::transpose` rejects **rank ≥ 5** → `pixel_shuffle` (389), `pixel_unshuffle` (368),
  rank-5 `transpose_copy`/`permute_copy` SKIP.
- `cortex_m::minimum` (34) / `cortex_m::quantized_mul` (10) **hard-abort** (`ET_CHECK` in
  `cortex_m_ops_common.h`) when the pass routes non-int8 operands into the int8 CMSIS kernel — a
  lowering bug (rewrite fired without ensuring int8 operands).

## Not filed here
- **0 device crashes** across 61,735 graphs. All 253 "CRASH" are runner **build** failures
  (`linear`/`linear.out`/`_fft_r2c`, rc=5); all 419 TIMEOUT are `_fft_r2c` build/run hangs — infra,
  no device verdict (see [../skips.md](../skips.md)).
- Portable-compute mismatches (`index_put`, `fill.Scalar`, bitwise shifts = UB, `remainder`, …) and
  portable-kernel SKIPs (int64-index, conv rank, group_norm, …) → [../ruled_out/](../ruled_out/).
