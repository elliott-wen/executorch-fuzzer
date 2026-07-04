# ethos-u `fill.Scalar` — fill value ignored, output is garbage (WRONG-VALUE)

- **Failure mode:** MISMATCH
- **Mechanism:** WRONG-VALUE — the delegate ignores the fill scalar entirely; the output is a
  fixed garbage pattern derived from the *input* buffer, independent of the requested fill value.
- **Bucket:** delegated (ops≥1) — Ethos-U delegate kernel. Confirmed on the Corstone-300 FVP.
- **Determinism:** REPRO 5/5 (deterministic). Domain: FINITE-CLEAN (leaf finite).
- **Prevalence:** 96 of 100 delegated `fill.Scalar` samples MISMATCH.
- **Repro:** `bugs/repro_fill-scalar.py` (job `w0:337`)

## Device-verified evidence
`n0 = fill.Scalar(L0, value)` fills a tensor of `L0`'s shape with a scalar. The reference is
constant (all = fill value); the device returns the **same non-constant pattern every time**,
regardless of the fill value or `L0`:

| job | fill value (eager) | eager out | device out |
|-----|--------|-----------|------------|
| w0:337   | 1 | `[1,1,1,1]`                     | `[-3,-1,-3,-1]` |
| w102:362 | 1 | `[1,1,1,1,1,1,1,1,…]`           | `[-3,-1,-3,-1,-3,-1,4,-4,…]` |
| w104:327 | 3 | `[3,3,3,3,3,3,3,3,…]`           | `[-3,-1,-3,-1,-3,-1,4,-4,…]` |

The device output `[-3,-1,-3,-1,4,-4,…]` is **identical across jobs with different fill values**
→ the fill scalar is never applied; the kernel emits stale/input-derived data. (Output dtype is
also downcast int64→int32, but the value error stands independent of dtype.)

## Reproduce
```
# with a broker + FVP ethos-u client running (see _repro_common.py header)
python findings/ethos_u/bugs/repro_fill-scalar.py
```
