# CoreML `avg_pool2d` — wrong divisor (≈2× values) + static-resize SKIP + rare crash

- **Failure mode:** MISMATCH (primary), plus SKIP and a rare CRASH form
- **Root cause:** operator kernel (delegated to Core ML; `delegated.ops=1`, `non_delegated=0`)
- **Mechanism:** WRONG-VALUE — averaging uses the wrong window/divisor (consistent with a
  `count_include_pad` / `ceil_mode` mismatch), giving roughly a constant factor error
- **Occurrences:** 10 delegated MISMATCH + 217 SKIP + a rare crash. MISMATCH device-verified,
  deterministic (5/5).
- **Repro:** [repro_avg_pool2d.py](repro_avg_pool2d.py) (jobs `w104:770`, `w117:97`)

## What happens
The pooled averages are off by (mostly) a factor of ~2 vs the reference, with finite in-distribution
inputs:

```
job w104:770   avg_pool2d(L0: float32, finite)
  eager : [-0.120623, 0.511092, 0.290897, 0.297189]
  device: [-0.241211, 1.022461, 0.582031, 0.594238]     ~2x eager  (max|delta|=0.511)
```

Separately, most `avg_pool2d` forms **SKIP** at runtime with
`[tensor_impl.cpp:140] Attempted to resize a static tensor. Expected shape (a,b,c), but received
(a,b',c)` — the delegate builds a static output shape that disagrees with the runtime shape (see
[../skips.md](../skips.md)).

## Why
The ~2× factor points at the pooling divisor: PyTorch's default `count_include_pad=True` divides by
the full kernel area, and `ceil_mode`/padding change which cells are counted. The Core ML lowering
appears to divide by a different count (e.g. valid cells only, or a mis-sized window), so the mean is
scaled. The dominant SKIP is a separate output-shape/resize mismatch in the same op's lowering.
