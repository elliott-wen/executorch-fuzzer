# SCALE (device ≈ eager × r) cluster — mostly a var.correction OPERATOR bug, not graph-opt

48 corpus cases classified `SCALE:r` (COMPOSITIONAL/SIBLING_DEPENDENT). Digging in, the cluster
splits into **one dominant, root-caused operator bug + a heterogeneous tail**:

## 28/48 — `var.correction(correction=None)` operator bug (the ×0.5 / ×0.75 sub-cluster)
Factor is exactly **(N−1)/N** (×0.5 at N=2, ×0.75 at N=4, …). Root cause: **CoreML computes the
biased variance (÷N) for `correction=None` where PyTorch means unbiased (÷(N−1))**. It read as
COMPOSITIONAL only because the bisector baked var's correct value; `var.correction` **alone**
mismatches → it's an **operator bug**, propagated as a scale onto downstream consumers
(`div`/`mul`/`select`/`sinh`/…). Full analysis + minimal repro:
[operator_var_correction.md](operator_var_correction.md). **This is the SCALE headline.**

## 20/48 — heterogeneous tail (separate per-case triage)
Varied factors and triggers, no single mechanism — candidates only until individually verified:

| factor | job (example) | trigger | likely nature |
|---|---|---|---|
| ×−1 | w107:89, w49:253 | sign.out / cosh | sign handling |
| ×2, ×3 | w38:323, w29:492, w47:599 | mul/fill, transpose, prod | integer / reduction |
| ×2.31e18, ×6.87 | w56:665, w33:123 | bitwise_left_shift, upsample | overflow-adjacent |
| ×0.056, ×0.0782, ×0.368 | w83:300, w27:300, w12:316 | elu, gelu, copy | needs pin |
| ×0.75, ×0.857, ×0.901 | w101:701, w16:40, w40:761 | upsample, any.dims, bitwise_xor | needs pin |

These are a minority and not one bug — each needs the same "run the trigger/target alone" check the
var case got. Not filed as confirmed until pinned.

## Takeaway
The SCALE mechanism is **not** primarily a graph-optimization bug — the bulk (28) is the
`var.correction` operator bug surfacing through its consumers. This is the workflow's confound
lesson in action: a "diverges only with an upstream/sibling op" (COMPOSITIONAL/SIBLING) verdict
must be re-checked by running that op alone before calling it graph-opt.
