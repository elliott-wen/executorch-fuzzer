# XNNPACK arm64 `logit` — saturates `+inf` output to ~88.38

- **Mode:** MISMATCH (non-finite) · **Root cause:** operator kernel (delegated, `ops≥1`)
- **Arch:** arm64 / NEON **only** · **Occurrences:** 187 (`logit.out` 110 + `logit` 77)
- **Repro:** [repro_xnnpack-logit-nonfinite.py](repro_xnnpack-logit-nonfinite.py) (`w0:708`)

```
eager (ATen)     : [-inf, inf]
device (arm64)   : [-inf, 88.375]
```

`logit(x) = log(x/(1-x))` — the arm64 NEON `log` component saturates `+inf` to ~88.375 (same constant
as the `log` bug), while `-inf` is preserved. Shares the root cause with
[xnnpack-log-saturate.md](xnnpack-log-saturate.md).
