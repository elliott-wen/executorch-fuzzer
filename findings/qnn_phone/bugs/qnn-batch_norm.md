# QNN `_native_batch_norm_legit.no_stats` — grossly wrong output

- Mode: MISMATCH (value) · delegated · phone-verified · 166 (164 value)
```
eager: [-1, 1, 1, 1, 1, -1, -1, -1]   phone: [-15256, 35104, 29536, 36800, ...]
```
Batch-norm on the HTP produces values ~4-5 orders of magnitude wrong (Δ≈3.7e4) — the normalization is
broken, not precision.
