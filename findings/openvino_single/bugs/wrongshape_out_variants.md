# WRONG-SHAPE — reduction/topk `out=` variants don't resize the output tensor

**Failure mode:** MISMATCH (shape) · **Mechanism:** WRONG-SHAPE · **Bucket:** delegated · **Determinism:** REPRO 8/8.

## What happens
The `out=` forms of the multi-output reductions/selection ops return a **degenerate empty output**
on device where the reference is a small tensor:
```
job w0:115  min.dim_min(L0, dim, keepdim, out_values=L1, out_indices=L2)
eager  shape (1,)          device shape (0,0,0,0,0)
job w10:66  max.dim_max(...)
eager  shape (1,1)         device shape (0,0,0,0)
job w0:527  topk.values(...)
eager  shape (1,1)         device shape (0,0,0,0,0)
```
The preallocated `out=` tensors arrive as **empty placeholder leaves** (e.g. `L1`,`L2` with shape
`(0,0,0,0,0)`). PyTorch resizes them to the correct output shape; the OpenVINO delegate leaves them
at their empty preallocated shape → the device output is empty/degenerate.

## Affected operators
`min.dim_min` (`min.dim`), `max.dim_max` (`max.dim`), `topk.values`.

## Mechanism / severity
A meta/shape-inference gap in the OpenVINO lowering of `out=` reduction/selection variants: the
delegate does not resize preallocated output tensors. **Form-specific** — it is triggered by the
`out=` variant carrying empty preallocated out-tensors as graph leaves (a single-op-corpus shape),
so severity in real graphs (where out tensors are pre-sized) may be lower. Reported as a real
WRONG-SHAPE finding with this caveat.

## Repro
```
PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/bugs/repro.py w0:115 w10:66 w0:527
```
