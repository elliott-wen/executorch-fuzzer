# WRONG-VALUE — view/gather transform ignored (raw input returned)

**Failure mode:** MISMATCH · **Mechanism:** WRONG-VALUE · **Bucket:** delegated · **Determinism:** REPRO 8/8.

## What happens
For strided-view / gather ops, the OpenVINO delegate returns the **raw input data in its original
row-major order**, ignoring the op's stride/offset/window transform. Output *shape* is correct;
the *values* are the wrong elements. (Contrast with the ZEROED family, which drops the store
entirely — here the store lands but the transform never runs.)

### `diagonal_copy` — returns the first row, not the diagonal  (w0:329)
```
op    : diagonal_copy(L0, ...)     input 3x3
eager : [-0.482491,  0.513757, -0.79292 ]   <- the diagonal
device: [-0.482491,  1.110353,  0.934016]   <- row 0 of the input
```
`device[0]==eager[0]` only because `[0,0]` is both the first row element and the first diagonal
element; the rest are the wrong (row-major) elements.

### `unfold_copy` — returns input in row-major order, not the unfolded windows  (w0:220)
```
op    : unfold_copy(L0, ...)       input (2,4) -> output (1,4,2)
eager : [-0.482491, 0.513757, 1.110353, 0.674999, 0.934016, 0.102609, ...]  <- unfolded (strided)
device: [-0.482491, 1.110353, 0.934016, 1.163586, 0.513757, 0.674999, ...]  <- raw input order
```
Same multiset of values, wrong ordering → the unfold windowing/stride was not applied.

## Mechanism
The delegate treats these as a reshape/passthrough of the input buffer and emits the contiguous
input bytes with the output's metadata, rather than applying the gather implied by the view's
strides/offset. A genuine per-op lowering bug (wrong data, right shape).

## Repro
```
PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/bugs/repro.py w0:329 w0:220
```
