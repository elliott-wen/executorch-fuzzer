# Moto G54 (Mali-G57): broad fp16 precision drift (matmul + layout/copy carriers)

**Signature:** `delta` (loose & strict) across many ops — `linear`/`bmm` born-here; `diagonal_copy`/
`pixel_shuffle`/`view_copy`/`squeeze_copy`/`permute_copy`/`transpose_copy`/`slice_copy` as flagged
**carriers**.
**Cluster:** ~900 of the 1,200 Moto-only mismatches are deltas (607 delta_loose + 297 delta_strict);
the rest are the special-value ops above.
**Classification:** **device-specific (budget Mali precision)** — not a delegate bug; the Mali-G57
computes in looser/lower precision so accumulated fp16 rounding crosses the eager-fp32 tolerance on
far more graphs than the flagship GPUs.

## What the localizer shows: 80% inherited

The on-device first-divergence localizer (Moto attached, 75 jobs localized) finds **80% of Moto-only
mismatches are *inherited*** — the flagged output op is **not** the born-here root. The layout/copy
ops that dominate the flagged-op table are mostly **carriers**: they copy/reshape a value that
already drifted upstream. Examples (flagged → born-here root):

```
slice_copy   -> slice_copy / pow / alias_copy      pixel_unshuffle -> expand_copy / pixel_shuffle
diagonal_copy-> diagonal_copy / _log_softmax / sin  squeeze_copy    -> squeeze_copy / lift_fresh_copy / linear
split_copy   -> round / linear / expand_copy        mul             -> pow / gelu / _log_softmax
```

The genuine born-here delta roots are **`linear`** (matmul, 6 in the sample), with smaller
contributions from `gelu`, `leaky_relu`, `constant_pad_nd`, `_log_softmax`, `slice_copy`,
`expand_copy`. Sample detail (eager `e` ≈ Moto `b`, just past tolerance):
`w0:1846` linear `e=3 b=3`, `w0:213` linear `e=-2 b=-2`, `w29:716` gelu `e=2.512 b=2.513`,
`w36:1434` _log_softmax `e=-2.738 b=-2.738`.

## Over-representation (what the budget GPU drifts on most)

Ops most over-represented in Moto-only vs the shared bugs (lift vs the device-invariant set):

| op | lift | role |
|---|---:|---|
| `diagonal_copy` | 14.1× | carrier (reshape of a drifted value) |
| `pixel_shuffle` | 10.1× | carrier / layout arithmetic |
| `view_copy` | 6.0× | carrier |
| `linear` / `bmm` | (strict) / 4.9× | **born-here** — matmul accumulation in lower precision |
| `squeeze_copy`/`permute_copy`/`transpose_copy`/`select_copy` | 2.9–3.7× | carriers |

## Mechanism

Two compounding effects of the budget part:
1. **Lower-precision matmul** (`linear`/`bmm`): the Mali-G57 accumulates GEMMs in fp16 (or a shorter
   accumulator) so results land just outside the eager-fp32 tolerance — born-here deltas.
2. **fp16 storage round-trips through layout ops**: every reshape/copy re-materializes fp16 values;
   accumulated rounding crosses the comparison threshold *at* a copy node, so first-divergence (a
   per-node tolerance check) attributes it there — but it's inherited drift, not a copy bug.

This is why the Moto's fringe is broad (many op families) and ~75% precision deltas: it's the
GPU-tier precision axis, not a per-op defect. The flagship Pixel 9 (Mali-G715) shows almost none of
this (~20 device-only total), confirming it's the *budget* tier, not Mali per se.

## Evidence
`tmp/run_vulkan_motog54/moto_localized.jsonl` + `moto_localized_aggregate.txt` (on-device localizer);
`compare_6way.txt` (lift table). Caveat: matmul/precision born-here roots are real; the copy-op
flags are carriers (80% inherited).
