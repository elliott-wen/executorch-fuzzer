# NONFINITE — `_native_batch_norm_legit.no_stats` emits nan (finite leaf + finite reference)

**Failure mode:** MISMATCH · **Mechanism:** NONFINITE · **Bucket:** delegated · **Determinism:** REPRO 8/8.

## What happens
```
job w106:221  _native_batch_norm_legit.no_stats(L0, ...)   input (1,2) fp32
leaf  : [-0.482491, 1.110353]   FINITE
eager : [0.0, 0.0]              FINITE   (PyTorch: batch of 1 -> zero variance -> normalized 0)
device: [nan, nan]              <- OpenVINO divides by zero std
```
Leaf **finite**, reference **finite** — the device introduces the nan. Sampling 12 nonfinite
`_native_batch_norm_legit.no_stats` jobs: **12/12 have a finite eager reference** (device-introduced
nan), so this is a genuine op bug, not the reference-nan confound that afflicts `native_group_norm`
(see `ruled_out/native_group_norm_reference_nan.md`).

## Mechanism
With a batch dimension of 1 (or otherwise zero variance) the OpenVINO batch-norm kernel computes
`x / sqrt(var + eps)` without the eps guard actually preventing the divide-by-zero (or with eps not
applied), yielding nan. PyTorch produces 0. fp16 variants also overflow to nan.

## Repro
```
PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/bugs/repro.py w106:221
```
