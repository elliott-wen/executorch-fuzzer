# XNNPACK `gelu` — `gelu(inf)` returns `inf`, ATen returns `NaN` (shared x86 + arm64)

- **Mode:** MISMATCH (non-finite) · **Root cause:** operator kernel (delegated, `ops≥1`)
- **Arch:** both arm64 and x86 · **Occurrences:** 92 delegated (arm64)
- **Repro:** [repro_xnnpack-gelu-nonfinite.py](repro_xnnpack-gelu-nonfinite.py) (`w1:776`)

```
eager (ATen)     : [nan, nan]
device (arm64)   : [inf, inf]
```

`gelu(inf)`: the kernel passes `+inf` through (`inf·Φ(inf)=inf`) where ATen returns `NaN`. Reproduces
identically on x86 and arm64 — a backend-level (not arch-specific) bug.
