# RULED OUT — multi-output `out=` output misalignment (NOT a CUDA bug)

**Affects:** `min.dim_min`, `max.dim_max`, `topk.values` — the MISMATCH "shape/rank divergence"
class (304) plus the handful of `topk` "wrong-value" cases (~16). ~320 MISMATCHes.

## Symptom
The feeder reports e.g. `out[0] shape (1,) vs (0,0,0,0,0)` — the device's user output looks empty.

## Why it is not a device bug (device computes the correct answer)
These ops return **multiple outputs** (values + indices) via `out=` params, so the `.pte` returns
several tensors. The harness selects the user output by `user_pos` (from the exported program's
`output_specs`). For `min.dim` job **w0:115** (eager values shape `(1,)`, `user_pos=[2]`):

| backend | all device outputs | device[user_pos=2] | correct? |
|---|---|---|---|
| **portable** | `[(0,0,0,0,0), (1,), (1,)]` | `(1,)` | ✅ aligned |
| **xnnpack** | `[(0,0,0,0,0), (1,), (0,0,0,0,0)]` | `(0,0,0,0,0)` | ❌ empty |
| **cuda** | `[(0,0,0,0,0), (1,), (0,0,0,0,0)]` | `(0,0,0,0,0)` | ❌ empty |

The **correct value `(1,)` is present at device index 1** on cuda — the kernel is right; `user_pos`
just points at the wrong slot. And it reproduces identically on **xnnpack**, so it is **not
CUDA-specific** — it is a general delegate ⇄ `output_specs` ordering mismatch for multi-output `out=`
ops. Per the analysis workflow (step 5: drop/realign prepended mutation outputs), these are a
harness/oracle-alignment confound, not a backend operator bug.

## Follow-up (separate from CUDA)
Worth a harness fix (align `user_pos` to the delegate's actual output order for multi-output `out=`
ops) and possibly a real ExecuTorch delegate-output-ordering issue — but it is **not** a CUDA
finding and is excluded from the confirmed bugs.
