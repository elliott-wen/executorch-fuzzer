# gelu / floor_divide / grid_sampler_2d / elu (fp16) — NONFINITE divergence

- **Failure mode:** MISMATCH (non-finite positions differ) · **delegated** · deterministic 5/5
- **Leaf check (3e):** leaf inputs **finite** → the divergence is produced by the op, not inherited.
- **Mechanism:** NONFINITE — at an fp16 overflow/degenerate point the CUDA kernel and the CPU
  reference disagree on **`NaN` vs `Inf`** (not a wrong finite value; both go non-finite, differently).

## Device-verified evidence (all float16)
| job | op | eager (CPU) | device (GPU) |
|---|---|---|---|
| w0:117 | `gelu.out` | `[NaN, NaN]` | `[Inf, Inf]` |
| w11:273 | `floor_divide.out` | `[NaN, NaN]` | `[Inf, Inf]` |
| w10:170 | `grid_sampler_2d` | `[Inf]` | `[NaN]` |

~134 finite-leaf cases across these ops (`gelu` 65, `floor_divide(.out)` 45, `grid_sampler_2d` 16,
`elu.out` 8). The remaining non-finite MISMATCHes are **not** in this bucket: 136 had a non-finite
leaf (filtered, 3e) and ~306 are `_native_batch_norm_legit` where the **CPU reference** is the
degenerate/non-finite side (see [../ruled_out/batchnorm_reference_degenerate.md](../ruled_out/batchnorm_reference_degenerate.md)).

## Severity
Soft: fp16 overflow semantics are not tightly specified and both results are non-finite, so this is
a low-severity divergence rather than a wrong finite value. Reported as a class for completeness.

## Reproduce
```bash
... .venv-cuda/bin/python findings/cuda_h200/bugs/repro.py w0:117
```
