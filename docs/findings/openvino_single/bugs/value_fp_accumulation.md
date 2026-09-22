# VALUE — matmul/conv fp32 accumulation (low severity, near-tolerance)

**Failure mode:** MISMATCH · **Mechanism:** WRONG-VALUE (precision) · **Bucket:** delegated ·
**Determinism:** REPRO 8/8.

## What happens
The GEMM/conv family diverges from the CPU reference by a small, consistent amount just beyond the
oracle's `rtol=0.01`:
```
addmm.out    max|delta| ~ 5e-3 .. 2.4e-2
bmm.out      value (8/8)
convolution  value (8/8)   (also a rare native abort — see crash doc)
linear/linear.out  value (2/2)
```
These are **fp32 accumulation-order differences** on the OpenVINO CPU backend (OpenVINO reorders the
reduction / uses different blocking than the reference), consistent with the prior observation that
OpenVINO CPU conv matches eager only to ~5e-3 with cosine ≈ 0.99999.

## Severity
Low. They exceed the project oracle's tolerance and are deterministic, so they are reported as
MISMATCH, but the mechanism is numerical accumulation, not a logic bug. Treat as precision-class /
candidate-for-tolerance rather than a functional defect. Listed here for completeness; not promoted
to a hard bug.
