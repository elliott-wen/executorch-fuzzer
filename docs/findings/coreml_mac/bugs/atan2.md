# CoreML `atan2` — `atan2(0, x<0)` returns `0` instead of `π`

- **Failure mode:** MISMATCH
- **Root cause:** operator kernel (delegated to Core ML; `delegated.ops=1`, `non_delegated=0`)
- **Mechanism:** WRONG-VALUE — the `y==0, x<0` branch of `atan2` returns `0` instead of `π`
- **Occurrences:** 53 delegated mismatches. Device-verified, deterministic (5/5 re-runs identical).
- **Repro:** [repro_atan2.py](repro_atan2.py) (corpus jobs `w100:421`, `w102:358`)

## What happens
For input elements where `y == 0` and `x < 0`, the correct `atan2(0, x<0) = π`. The Core ML
delegate returns `0.0`. Elements with `x > 0` or `y != 0` match the reference.

```
job w100:421   atan2(L0=[True,False] as y, L1=[neg, pos] as x)   float32 out
  eager : [3.141593, 0.244979]      # atan2(1,-x)=π ,  atan2(0? ...)
  device: [0.0,      0.244995]      # π collapsed to 0
job w102:358
  eager : [3.141593, 3.141593, 0.0, 2.819842, 0.463648, ...]
  device: [0.0,      0.0,      0.0, 2.818359, 0.463623, ...]
```

Both leaf inputs are finite (checked): the wrong value is manufactured by the kernel, not the input.

## Why
`atan2(y, x)` for `y == 0` is defined piecewise: `+0`/`-0` for `x ≥ 0`, `±π` for `x < 0`. The Core ML
lowering appears to implement `atan2` via `atan(y/x)` (which gives `atan(0) = 0`) without adding the
`x < 0` quadrant correction of `±π`. So the `x < 0, y == 0` boundary — and the negative-x half-plane
generally when it needs the `±π` offset — is wrong.

## Scope
Triggered by ordinary finite inputs (a negative `x` with a zero `y`), so it is in-distribution and
reproduces readily. `max|delta| = π` (3.1416) — a full-magnitude error, not a rounding artifact.
