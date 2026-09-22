# CoreML `_upsample_bilinear2d_aa` — antialiasing ignored (wrong values)

- **Failure mode:** MISMATCH
- **Root cause:** operator kernel (delegated to Core ML; `delegated.ops=1`, `non_delegated=0`)
- **Mechanism:** WRONG-VALUE — the anti-aliased bilinear resize is computed as if `antialias=False`
  (and/or with a different sampling grid)
- **Occurrences:** 77 delegated mismatches. Device-verified, deterministic (5/5).
- **Repro:** [repro_upsample-bilinear2d-aa.py](repro_upsample-bilinear2d-aa.py) (jobs `w0:195`, `w1:294`)

## What happens
`_upsample_bilinear2d_aa` is the **anti-aliased** bilinear resize (the `aa` variant applies a wider
support filter when downsampling). The Core ML output diverges from the reference by a large margin —
consistent with the delegate performing a plain (non-antialiased) bilinear sample:

```
job w1:294   _upsample_bilinear2d_aa(L0: float32, finite)
  eager : [0.090424, 0.090424, 0.179231, 0.268038, 0.467..]
  device: [-1.158203, -0.827637, -0.496826, -0.166016, 0..]      max|delta| = 2.334
job w0:195
  eager : [-0.482491, -0.482491, 0.048457, 0.579405, ...]
  device: [-0.482422,  0.048340, 0.579590, 1.110352, ...]        max|delta| = 0.531
```

`max|delta|` up to **2.33** on ~unit-scale data — far outside any tolerance; the sampled positions
and/or the antialias weighting are wrong, not a rounding issue.

## Why
Core ML's resize does not carry PyTorch's antialias filter semantics; the delegate maps the `_aa`
op onto its ordinary bilinear resize. When the op is genuinely downsampling (where `antialias`
changes the result), the output is wrong. Inputs are finite and in-distribution.
