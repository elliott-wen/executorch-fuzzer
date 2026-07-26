# CoreML precision losses — `var`/`batch_norm`/`logit`/`pow`/`conv` beyond tolerance

- **Failure mode:** MISMATCH (small, consistent)
- **Root cause:** operator kernel (delegated to Core ML; `delegated.ops≥1`)
- **Mechanism:** WRONG-VALUE (precision) — the value is close but exceeds the per-dtype tolerance
  (`rtol=0.01, atol=0.001`), consistently, on finite inputs. These are lower-severity than the
  full-magnitude bugs above but are **not** flaky (5/5) and **not** dtype artifacts.
- **Repro:** [repro_precision-reductions.py](repro_precision-reductions.py)

| operator | example Δ | dtype | job | note |
|----------|-----------|-------|-----|------|
| `var.correction`     | 0.064 | float32 | `w1:284` | variance reduction precision |
| `var.correction_out` | 0.153 | float16 | `w0:93`  | worse in fp16 |
| `_native_batch_norm_legit.no_stats` | 0.019 | float32 | `w0:20` | normalize precision |
| `logit`              | 0.126 | float32 | `w0:649` | large near the 0/1 asymptotes (nan positions match) |
| `pow.Scalar_out`     | 0.037 | float32 | `w10:755` | small-base precision |
| `remainder.Tensor_out` | 0.066 | float32 | `w74:40` | (cf. the ZEROED `remainder.Scalar` bug) |
| `convolution`        | 0.006 | float32 | `w100:93` | marginal — right at the tolerance edge |

```
job w1:284   var.correction(L0: float32)
  eager : [1.029844]      device: [0.965820]      max|delta| = 0.064
job w0:93    var.correction_out(L0: float16)
  eager : [0.611816]      device: [0.458984]      max|delta| = 0.153   (fp16)
```

## Why
Reductions (`var`, `mean`-of-squares inside batch_norm) accumulate in a lower-precision or
differently-ordered arithmetic on the delegate; fp16 (`var.correction_out`) is visibly worse. `logit`
and `pow` lose precision near their singular regions. `convolution` is borderline (could be argued
in-tolerance). Reported for completeness and separated from the full-magnitude WRONG-VALUE bugs.
