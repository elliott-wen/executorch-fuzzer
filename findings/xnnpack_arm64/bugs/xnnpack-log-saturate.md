# XNNPACK arm64 `log` — saturates `log(inf)` to ~88.38 instead of `inf`

- **Mode:** MISMATCH (non-finite) · **Root cause:** operator kernel (delegated, `ops≥1`) · **Gate:** 5/5 stable
- **Arch:** arm64 / NEON **only** — does NOT reproduce on x86 XNNPACK
- **Occurrences:** 180 delegated (complete, after re-running dropped jobs) · **Repro:** [repro_xnnpack-log-nonfinite.py](repro_xnnpack-log-nonfinite.py) (`w0:544`)

```
eager (ATen)     : [inf, inf]
device (arm64)   : [88.37626, 88.37626]
```

The arm64 NEON `log` polynomial **saturates** its output at ~88.376 (≈ `ln(FLT_MAX)`) for a `+inf`
input instead of returning `+inf`. The saturation constant is the kernel's max-range clamp leaking
into the non-finite domain. Deterministic (5/5).
