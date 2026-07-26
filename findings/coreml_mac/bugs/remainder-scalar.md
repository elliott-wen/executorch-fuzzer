# CoreML `remainder.Scalar` — output all-zeroed

- **Failure mode:** MISMATCH
- **Root cause:** operator kernel (delegated to Core ML; `delegated.ops=1`, `non_delegated=0`)
- **Mechanism:** ZEROED — the whole output is `0` where the reference is a nonzero remainder
- **Occurrences:** 3 delegated mismatches (`remainder.Scalar`) + 1 (`remainder.Tensor_out`, a small
  precision Δ — see README table). Device-verified, deterministic (5/5).
- **Repro:** [repro_remainder-scalar.py](repro_remainder-scalar.py) (corpus jobs `w14:186`, `w35:750`)

## What happens
`remainder(x, s)` for a finite tensor `x` and scalar divisor `s` should return the elementwise
remainder. The Core ML delegate returns **all zeros**.

```
job w14:186   remainder.Scalar(L0, s)   float32, leaf finite
  eager : [0.84206, 0.833971, 0.306189, 0.973745, 0.188222, 0.074766, 0.526731, 0.998707]
  device: [0.0,     0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0]
```

## Why
`remainder` lowers to `x - s * floor(x / s)`. The all-zero result is consistent with the delegate
dropping/zeroing one operand of that expression (e.g. computing `x - x`, or a scale/store that lands
on zero), so every element collapses to `0` regardless of `x`. A ZEROED output = a dropped store or a
dropped operand in the op's own lowering.
