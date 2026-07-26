# Ruled out — `native_group_norm` "nonfinite" is a reference-nan confound (step 3d/3e)

`native_group_norm` produced 388 "non-finite mismatch" outcomes on the delegate. Before filing them
as a device NONFINITE bug, the leaf-and-reference sanity check (step 3e) shows most are the
**reference** being nan, not the device:

```
job w0:139  native_group_norm(L0, ...)   input (2,2) fp16
leaf  : [-0.482, 1.110, 0.934, 1.164]   FINITE
eager : [nan, nan, nan, nan]            <- PyTorch reference is itself nan (degenerate group)
device: [-1.0, 1.0, -1.0, 1.0]          FINITE
```

Sampling 12 `native_group_norm` nonfinite jobs: **10/12 have an eager reference that is itself
non-finite** (the device output is *finite*). PyTorch's group-norm goes nan on these degenerate
single-op shapes (tiny group / zero variance); OpenVINO returns a finite normalized value. The
oracle flags the *differing* nan positions, but the nan originates in the **reference**, so this is
a reference-side confound, **not** a device operator bug.

The other **2/12** have a finite reference and could be genuine fp16 device nonfinite — a small
residual; not promoted without a larger confirmed sample. Contrast with
`_native_batch_norm_legit.no_stats`, where **12/12** have a finite reference and the device
introduces the nan (that one is a confirmed bug — see `bugs/nonfinite_batchnorm.md`).
