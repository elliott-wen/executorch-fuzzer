# `native_group_norm` — portable kernel emits `NaN` in different positions than ATen

- **Mode:** MISMATCH (non-finite) · **Occurrences:** 17 · **Repro:** [repro_native_group_norm.py](repro_native_group_norm.py) (job `w0:72`)

```
eager (ATen)    : [nan, nan, nan, nan, nan, nan]
device (portable): [nan, 0.0, 0.0, nan, 0.0, nan]
```

On a degenerate group (zero/near-zero variance, or a non-finite element), ATen normalizes to all-`NaN`
while the portable kernel produces a *mix* of `NaN` and `0.0` — the two disagree on which elements go
non-finite (`(x-μ)/√(σ²+ε)` handled differently at the degenerate boundary). Divergence confirmed on
device.

(`native_group_norm` is also the largest SKIP class — 348 — a separate coverage gap where the
portable kernel rejects the graph at runtime.)
