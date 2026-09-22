# XNNPACK `rsqrt` — `rsqrt(0)` returns `NaN`, ATen returns `inf`

- **Mode:** MISMATCH (non-finite) · **Root cause:** operator kernel (delegated, `ops≥1`) · device-verified
- **Repro:** [repro_xnnpack-rsqrt-nonfinite.py](repro_xnnpack-rsqrt-nonfinite.py)

```
eager (ATen)     : [inf, 0.5]
device (xnnpack) : [nan, 0.5]
```

The fast reciprocal-sqrt has no zero-domain guard: `rsqrt(±0)` yields `NaN` (`0·(1/0)`-style) instead
of `+inf`. This is the `rsqrt` half of the sqrt/rsqrt domain family. **Corpus visibility:** delegated,
but triggers only on an exactly-`±0` input (2/5 of the injection choices), so the corpus under-samples
it — fed `rsqrt(0)` directly it reproduces every time.
