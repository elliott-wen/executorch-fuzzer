# XNNPACK `gelu` — returns `inf` for `inf` input, ATen returns `NaN`

- **Mode:** MISMATCH (non-finite) · **Root cause:** operator kernel (delegated, `ops≥1`)
- **Occurrences:** 92 · **Repro:** [repro_xnnpack-gelu-nonfinite.py](repro_xnnpack-gelu-nonfinite.py) (`w1:776`)

```
eager (ATen)     : [nan, nan]
device (xnnpack) : [inf, inf]
```

`gelu(inf)`: the XNNPACK kernel passes `+inf` through (`inf·Φ(inf)=inf`), while ATen returns `NaN`
(its formulation hits `inf·0`/`inf−inf`). Non-finite-domain divergence surfaced by leaf injection.
