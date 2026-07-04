# XNNPACK activation NaN-launder — `relu` / `hardtanh` return a finite bound for `NaN` input

- **Mode:** MISMATCH (non-finite) · **Root cause:** operator kernel (delegated, `ops≥1`) · **Gate:** 5/5 stable
- **Occurrences (87,921-graph injected run):** `relu` 2, `hardtanh` 2 (low — see rate note)
- **Repros:** [repro_xnnpack-relu-nonfinite.py](repro_xnnpack-relu-nonfinite.py) (`w70:755`),
  [repro_xnnpack-hardtanh-nonfinite.py](repro_xnnpack-hardtanh-nonfinite.py) (`w108:307`)

```
relu     eager: [nan, nan, nan, nan, nan, 0.0748, 2.527, nan]
         device:[0.0, 0.0, 0.0, 0.0, 0.0, 0.0748, 2.527, 0.0]
hardtanh eager: [nan, nan, nan, nan, nan, -3.0, -3.0, nan]   (clamp to [-8, hi])
         device:[-8.0,-8.0,-8.0,-8.0,-8.0,-3.0,-3.0,-8.0]
```

The XNNPACK `relu`/`hardtanh` lower to SIMD `fmax`/`fmin` clamps, which return the **non-NaN operand**
— so a `NaN` input is laundered to the clamp bound (`0.0` for relu, the lower bound `-8.0` for
hardtanh) instead of propagating as `NaN` (ATen). Confirmed only on the **fp16** SIMD path in this run
(an fp32 hand-test propagated NaN correctly).

**Rate note:** only caught when the injected leaf value is `NaN` (~1/5 of injections) AND the op is
delegated — effective trigger rate ≈1%, so counts are low despite the bug being deterministic (5/5).
