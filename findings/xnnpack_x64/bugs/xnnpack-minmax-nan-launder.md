# XNNPACK `minimum` / `maximum` — NaN-launder (return the finite operand for `min/max(NaN, x)`)

- **Mode:** MISMATCH (non-finite) · **Root cause:** operator kernel (delegated, `ops≥1`) · device-verified
- **Repros:** [repro_xnnpack-minimum-nonfinite.py](repro_xnnpack-minimum-nonfinite.py),
  [repro_xnnpack-maximum-nonfinite.py](repro_xnnpack-maximum-nonfinite.py)

```
minimum(NaN, 5.0): eager [nan, 2.0]  device [5.0, 2.0]
maximum(NaN, 5.0): eager [nan, 5.0]  device [5.0, 5.0]
```

`minimum`/`maximum` lower to SIMD `fmin`/`fmax`, which return the **non-NaN operand** — so a `NaN` in
one operand is laundered to the finite value instead of propagating (ATen returns `NaN`). Same
mechanism as the `relu`/`hardtanh` clamp launder ([xnnpack-activation-nonfinite.md](xnnpack-activation-nonfinite.md)).
**Corpus visibility:** delegated, but triggers only when a `NaN` lands in exactly the winning operand
(~1/5 × operand-position) so the corpus under-samples it — fed the exact trigger it reproduces every time.
