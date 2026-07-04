# XNNPACK `exp` — `exp(NaN)` returns `0.0`, ATen returns `NaN`

- **Mode:** MISMATCH (non-finite) · **Root cause:** operator kernel (delegated, `ops≥1`) · device-verified
- **Repro:** [repro_xnnpack-exp-nonfinite.py](repro_xnnpack-exp-nonfinite.py)

```
eager (ATen)     : [nan, 2.71828]
device (xnnpack) : [0.0, 2.71828]
```

XNNPACK's rational-polynomial `exp` has no NaN/overflow guard: `exp(NaN)` evaluates the polynomial to
`0.0` instead of propagating `NaN`. **Corpus visibility:** the op IS delegated, but this only triggers
on a `NaN` input (~1/5 of the uniform-random boundary injection), so the corpus under-samples it and
reports 0 hits — fed the exact trigger here it reproduces every time.
