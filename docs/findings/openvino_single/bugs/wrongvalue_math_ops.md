# WRONG-VALUE — math kernels (both sides finite, values differ beyond rtol/atol)

**Failure mode:** MISMATCH · **Mechanism:** WRONG-VALUE · **Bucket:** delegated · **Determinism:** REPRO.

Both eager and device outputs are finite and the same shape, but values diverge beyond the
per-dtype tolerance (rtol=0.01/atol=0.001 for float, exact for int).

## `reflection_pad2d` / `reflection_pad2d.out` / `reflection_pad1d.out` — wrong padding values
```
job w0:121  reflection_pad2d(L0)   input (2,4,4) fp16 -> (2,10,2)
eager : [-0.900391, 0.028809, 0.314209, -0.797363, -0.25708, 2.527344, ...]
device: [ 0.074768, 2.527344, -1.423828, -0.797363, 0.964355, 0.028809, ...]
```
The reflected border values are placed wrong — the reflection indexing in the OpenVINO lowering is
incorrect. (`reflection_pad1d.out` also produces ZEROED samples; `reflection_pad2d.out` also
produces rare native aborts — see crash doc.)

## Other confirmed WRONG-VALUE ops (delegated, REPRO 8/8 unless noted)
| operator | note |
|---|---|
| `remainder.Tensor_out` | integer/float remainder values differ (delta up to 4.0 on int) |
| `remainder.Scalar_out` | same, scalar RHS (2/2 sampled) |
| `pow.Scalar_out` | value divergence (7/7) |
| `var_mean.correction` | variance/mean output values differ (fp precision, 8/8) |
| `var.correction` | value (7/8) + occasional ZEROED (1/8) |

## Mechanism
Per-op kernel math error in the OpenVINO lowering (reflection indexing for the pads; the variance
ops are candidate fp-precision in the mean/variance reduction). Pinned input→output pairs above;
the reduction/variance ones are lower severity (precision-class).

## Repro
```
PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/bugs/repro.py w0:121
```
