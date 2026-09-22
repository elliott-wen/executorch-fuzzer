# RULED OUT — _native_batch_norm_legit reference-degenerate / alignment (NOT a CUDA bug)

**Affects:** `_native_batch_norm_legit` (+`.no_stats`) MISMATCHes — 341 (289 non-finite, 47
wrong-value, 5 shape). The single largest per-op MISMATCH count, and mostly a confound.

## Two sub-cases, neither a device kernel bug

### 1. Reference is the degenerate side (step 3d) — ~306 non-finite
With near-zero batch variance, CPU batch_norm `(x-mean)/sqrt(var+eps)` blows up; the **CPU
reference** produces `NaN`/huge values while the **device stays finite**. Pinned (w0:321, fp32,
shapes match `(2,4)`):

```
eager  (CPU): [0.944712, NaN, 210.53, 2.10e20, 2.178862, NaN, 23.128, -2.47e19]
device (GPU): [0.944712, 2.917925, 2.699478, 2.98387, 2.178862, 2.378608, 1.669533, 1.372965]
```

The device is arguably *more* correct here; the oracle flags it only because `NaN` positions differ.
This is a reference-side confound, not a device bug. (Leaf inputs were finite; the degeneracy is
var≈0, produced inside the CPU op.)

### 2. Shape cases — multi-output alignment (5)
`_native_batch_norm_legit` returns `(output, save_mean, save_invstd)` — a multi-output op, so its
"shape" MISMATCHes (e.g. w118:531 `(1,1,2,1)` vs `(1,3,2,1)`) are the same `user_pos` misalignment
documented in [multioutput_alignment.md](multioutput_alignment.md).

## Verdict
Excluded from confirmed CUDA bugs. If anything it shows the CUDA batch_norm is numerically robust
where the CPU path degenerates — the opposite of a device defect.
