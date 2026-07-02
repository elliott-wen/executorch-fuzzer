# XNNPACK `sqrt` domain error — `sqrt(negative) = -0.0` instead of `NaN`

- **Failure mode:** MISMATCH
- **Root cause:** operator kernel (delegated to XNNPACK; `delegated.ops≥1`)
- **Mechanism:** NONFINITE — missing `NaN` on the negative domain
- **Occurrences:** 92 delegated mismatches in this run (every `sqrt.out` mismatch; all "non-finite
  mismatch (nan/inf positions differ)")
- **Repro:** [repro_xnnpack-sqrt-domain.py](repro_xnnpack-sqrt-domain.py) (corpus job `w1:206`)

## What happens
For any negative input element, the XNNPACK `sqrt` kernel returns `-0.0`, where the eager CPU
reference (and IEEE semantics) returns `NaN`. Positive/zero elements match exactly.

```
job w1:206   input L0 = [<negative>, <positive>]   (float16)
  eager : [nan,  1.0537109375]
  device: [-0.0, 1.0537109375]     <- XNNPACK
```

Device-verified, deterministic (5/5 re-runs identical `-0.0`).

## Why
XNNPACK computes `sqrt(x)` via the fast reciprocal-sqrt identity `sqrt(x) = x · rsqrt(x)`. The
fast `rsqrt` has no negative-domain guard, so for `x < 0` the product collapses to `-0.0` instead of
producing `NaN`. This is the `sqrt` half of the documented sqrt/rsqrt domain family.

## Scope note
Caught because the trigger is an **in-distribution finite input** (a negative number, common in
random `randn` leaves) — the op itself manufactures the wrong non-finite result. The companion
`rsqrt(±0) ≠ ±inf` case was **not** triggered here (needs an exactly-`±0` input, which random leaves
essentially never produce). See the coverage section of the top-level [README](../README.md).
