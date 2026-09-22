# ZEROED (dropped store) — copy / view / identity ops ★ headline bug

**Failure mode:** MISMATCH · **Mechanism:** ZEROED (device output is all-zeros where the eager
reference is not) · **Bucket:** delegated (`ops≥1`) · **Determinism:** REPRO 8/8 (N=5 re-runs).

## What happens
Pure copy / view / identity operators, when the OpenVINO delegate absorbs them, return an
**all-zero output buffer** — the store into the output is dropped, so the delegate hands back the
zero-initialized allocation. Independent of dtype (float32, float16, bool, int64) and value.

Device-verified (`repro.py`):

| job | op | eager | device |
|---|---|---|---|
| w0:144 | `clone` (float32 [3,3]) | `[-0.482, 1.110, 0.934, 1.164, …]` | `[0,0,0,0,…]` |
| w0:375 | `clone` (bool [4]) | `[0,1,1,1]` | `[0,0,0,0]` |
| w0:260 | `alias_copy` (int64) | `[-3,-3,-3,4]` | `[0,0,0,0]` |
| w0:32  | `lift_fresh_copy` (int64) | `[-3,-3,-3,4,-3,-1,-4,-2]` | `[0,0,0,0,0,0,0,0]` |
| w0:183 | `t_copy` (float16) | `[-0.482, 1.110]` | `[0,0]` |
| w0:270 | `transpose_copy.int` (int64) | `[-3,-3]` | `[0,0]` |

## Affected operators (delegated, REPRO)
- **Pure ZEROED (8/8 samples):** `clone`, `alias_copy`, `lift_fresh_copy`, `t_copy`,
  `transpose_copy.int`.
- **Partially ZEROED** (some samples all-zero, others wrong-value depending on shape/dtype):
  `reflection_pad1d.out`, `rsub.Scalar`, `var.correction`, `var.correction_out`.

## Mechanism
These ops lower to an identity/passthrough or a trivial copy in the OpenVINO subgraph. The delegate
never writes the result into the ExecuTorch output tensor (the output buffer stays at its
zero-initialized state), so every element reads back 0. This is the classic *dropped store* — the
same class as the VGF/CoreML dropped-store findings, but here it hits the copy/view/identity family.

## Repro
```
PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/bugs/repro.py w0:144 w0:260 w0:183 w0:270
```
