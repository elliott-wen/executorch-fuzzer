# CoreML graph-optimization bug — deep mechanism investigation

**Bug:** the CoreML backend **drops the output store of a view/identity/copy op** whose output
tensor aliases a buffer the memory plan does not independently materialize. The returned tensor
reads back **all-zeros** (uninitialized). Kernel math is correct — it is a memory-planning defect,
triggered only in a multi-output graph.

- **Axis A:** MISMATCH · **Axis B:** GRAPH-OPTIMIZATION (memory plan) · **Mechanism:** ZEROED (dropped store)
- **Scope:** **204 of 503** bisected graph-opt cases (SIBLING_DEPENDENT + COMPOSITIONAL) — the single
  largest class; full corpus (10,518 graphs, 1,601 mismatches, 93.1% bisected). Every instance below
  is device-verified on the connected macOS CoreML worker; determinism-stable on replay.

---

## 1. It is CoreML-specific, not a harness/alignment artifact

The first thing ruled out (this exact trap fired on the CUDA run, where "dropped" outputs were
really a `user_pos` misalignment that reproduced on xnnpack). Here it does **not**: the identical
lowered subgraph is **correct on portable and xnnpack**, wrong only on CoreML.

Reduced repro `return (max(n2), lift_fresh_copy(n2))` (w1:8), all outputs dumped:

| backend | `user_pos` | out[0] = max | out[1] = lift_fresh_copy | eager out[1] |
|---|---|---|---|---|
| portable | [0,1] | `[-0.28]` ✓ | `[-0.28, -0.42]` ✓ | `[-0.28, -0.42]` |
| xnnpack | [0,1] | `[-0.28]` ✓ | `[-0.28, -0.42]` ✓ | `[-0.28, -0.42]` |
| **CoreML** | [0,1] | `[-0.28]` ✓ | **`[0.0, 0.0]`** ✗ | `[-0.28, -0.42]` |

Alignment is clean (`user_pos=[0,1]`, in order) and the correct value `-0.4206` is **absent from
every CoreML output** — it is genuinely dropped, not aliased/swapped.

## 2. Minimal root cause — two outputs aliasing one tensor (device-verified)

Hand-crafted minimal probes on CoreML (`n2 = x*2`), reporting which output zeroes:

| returned outputs | result |
|---|---|
| `lift_fresh_copy(n2)` alone | all correct |
| `lift_fresh_copy(n2), alias_copy(n2)` — 2 aliases of n2, **no other op** | **out[1] ZEROED** |
| `n2, alias_copy(n2)` — the tensor itself + one alias | **out[1] ZEROED** |
| `alias(L0), alias(L0)` — aliases of a **leaf** | **out[1] ZEROED** |
| `lift_fresh_copy(n2), alias_copy(n2), view_copy(n2)` — three aliases | **out[2] ZEROED** (last) |
| `alias(n2), alias(n3)` — aliases of **different** tensors | all correct (control) |
| `lift_fresh_copy(n2), alias_copy(n2)+0.0` — **neutralized** | **all correct** |

**Reading:** when ≥2 returned outputs alias the *same* underlying tensor, CoreML binds them to one
output buffer, writes it once, and the later aliases read zeros. No sibling *operator* is required —
a second *alias* is the trigger. Breaking the alias with a fresh compute (`+0.0`) fixes it — the
**A/B neutralization** that confirms it is the aliasing, not the kernel.

## 3. Lowered-IR evidence

`to_edge_transform_and_lower(CoreMLPartitioner)` on `(lift_fresh_copy(n2), alias_copy(n2))`:

```
call_function  executorch_call_delegate   args=[lowered_module_0, x]
call_function  getitem(delegate, 0)        → USER_OUTPUT
call_function  getitem(delegate, 1)        → USER_OUTPUT
output         (getitem, getitem_1)
```

Both aliasing ops are absorbed into **one** CoreML delegate returning two results; both are clean
`USER_OUTPUT`s at the ExecuTorch level. So the alias-to-one-buffer decision and the dropped store
happen **inside the compiled CoreML model** — consistent with portable/xnnpack (which don't delegate
this way) being correct.

## 4. The dominant corpus variant — view at a delegate boundary

Of the 50 corpus ZEROED cases, **1 is the pure two-alias case (§2); 49 are a single view/identity op
co-returned with non-alias siblings.** Device probes isolate the extra condition for that variant —
the view op's **source is a CoreML-unsupported op** (portable fallback ⇒ the aliased source isn't a
materialized delegate output):

| graph | source op | result |
|---|---|---|
| `n2 = div.Scalar_mode(x)` *(unsupported→portable)*; `return (max(n2), lift_fresh_copy(n2))` | boundary | **out[1] ZEROED** |
| `n2 = mul.Scalar(x)` *(supported→delegated)*; `return (max(n2), lift_fresh_copy(n2))` | none | all correct (control) |

This is why the corpus targets are dominated by view/copy/identity ops
(`lift_fresh_copy` ×10, `squeeze_copy` ×7, `alias_copy`/`view_copy`/`t_copy`/`detach_copy`) and why
`bisect_output_set` reported a *sibling* trigger (`max`, `prod`, `var.correction` ×15, …): the
sibling is simply whatever else makes the graph multi-output — the real trigger is the **aliased,
unmaterialized output buffer** of the boundary view op.

**Unified statement:** CoreML drops a returned tensor's store whenever that tensor aliases a buffer
the plan writes under a *different* name — either a second returned alias of the same tensor (§2) or
a view whose aliased source lives in a portable-fallback region (§4).

## 5. Confirmed instances (device-verified, target ← trigger)

| job | out | target op | trigger (`needed`) | eager → device |
|---|---|---|---|---|
| w1:8 | 3 | `lift_fresh_copy(n2=div)` | `max` | `[-0.28,-0.42]` → `[0,0]` |
| w1:781 | 1 | `lift_fresh_copy` | `n7` | `[-3,-3,-3,4]` → `[0,0,0,0]` |
| w1:489 | 5 | `lift_fresh_copy` | `prod` | `[-0.52,2.02,1.35]` → `[0,0,0]` |
| w101:421 | 3 | (copy-family) | `n3` | `[3.0]` → `[0.0]` |
| w103:475 | 1 | `alias`-family | `n3` | `[-0.48]` → `[0.0]` |
| w104:767 | 4 | `alias_copy` | `eq` | `[-0.50,1.35]` → `[0,0]` |
| w111:17 | 3 | `view_copy` | `var_mean.correction` | → `[0,…]` |
| w118:301 | 0 | `squeeze_copy.dims` | `abs` | → `[0,…]` |

(50 total in `graphopt_mechanisms.tsv`; `-0.0` appears in several, the uninitialized-buffer tell.)

## 6. Reproduce
```bash
# broker + Mac worker up.
# (a) MINIMAL self-contained two-alias repro — runs portable (correct) + coreml (out[1] zeroed):
BROKER_PORT=15554 PYTHONPATH=/data/jwen929 /data/jwen929/mobile/.venv/bin/python \
  /data/jwen929/mobile/findings/coreml_mac2/bugs/repro_alias_dropstore.py
# (b) corpus headline A/B (return target alone vs with sibling):
BROKER_PORT=15554 PYTHONPATH=/data/jwen929 /data/jwen929/mobile/.venv/bin/python \
  /data/jwen929/mobile/findings/coreml_mac2/bugs/repro_graphopt.py w1:8 3 n3
```
Verified output of (a):
```
[portable] out[0]=[3.08,-0.59,-4.36,1.14]  out[1]=[3.08,-0.59,-4.36,1.14]      # both correct
[coreml]   out[0]=[3.08,-0.59,-4.36,1.14]  out[1]=[0,0,0,0]  <-- ZEROED         # store dropped
```

## 7. Fix direction
CoreML lowering must **materialize a distinct output buffer for every returned tensor**, even when
two are views/aliases of the same source (or a view of a fallback tensor) — i.e. forbid
output-buffer aliasing across `USER_OUTPUT`s, or insert an identity copy when two outputs would
share storage. The `+0.0` neutralization (fresh buffer) is exactly this fix applied by hand.

## 8. Cross-backend check — does VGF have the *same* bug? (same class, different layer)

Ran the **identical minimal two-alias construct** on VGF (Vulkan lavapipe emulator), not borrowed
from VGF's report — a fresh device run:

| construct (`n2 = x*2`) | portable/xnnpack | **CoreML** | **VGF** |
|---|---|---|---|
| `lift_fresh_copy(n2)` alone | correct | correct | correct |
| `(lift_fresh_copy(n2), alias_copy(n2))` | both correct | **out[1] ZEROED** | **out[0] ZEROED** (3/3 deterministic) |
| `+0.0` on the ZEROED output | — | **fixes it** | **does NOT fix** |

**Same class, different layer** — the pure two-alias construct drops a store on **both** CoreML and
VGF (and on neither portable nor xnnpack, which share the ExecuTorch host planner), but:
- **CoreML:** the alias is on the **output binding inside the CoreML delegate** — `+0.0` on the
  output frees a fresh buffer and fixes it; portable/xnnpack (same host planner) are correct.
- **VGF:** the alias is on the **shared input buffer at the host memory planner** (the planner
  overlaps the output onto the input the VGF delegate consumes) — `+0.0` on the output does **not**
  help (must copy the *input* fresh), matching VGF's own plan-dump root cause.

So VGF "has the same issue" only in the sense of the same **failure class** (output-aliasing dropped
store) and the same triggering construct; the **root cause is a different defect in a different
layer**, exactly as `analysis.md` warns ("mechanisms differ per backend — don't assume it transfers").

## 9. SCALE cluster — dug in, and it's mostly an OPERATOR bug (var.correction)

The 48 `SCALE:r` cases are **not** primarily graph-opt. Digging in (device-verified):
- **28/48** are the **`var.correction(correction=None)` operator bug**: CoreML computes the
  **biased** variance (÷N) where PyTorch's `None` default means **unbiased** (÷(N−1)), so the result
  is off by exactly **(N−1)/N** (×0.5 at N=2, ×0.75 at N=4). Confirmed by running `var.correction`
  **alone** (mismatches by itself → operator bug), and CoreML-specific (portable/xnnpack correct).
  It was mislabeled COMPOSITIONAL because `bisect_cone` bakes var's correct value, so only the
  downstream consumer (`div`/`mul`/`select`/…) was the target and failed just with var live.
  → [bugs/operator_var_correction.md](bugs/operator_var_correction.md), repro `bugs/repro_var_correction.py`.
- **20/48** are a heterogeneous tail (×−1, ×2, ×3, ×0.056, ×2.3e18, …; triggers upsample/bitwise/
  gelu/elu/…) — no single mechanism, each needs its own "run the op alone" check; candidates, not filed.

**Method note (confound):** this is why `analysis.md` insists a COMPOSITIONAL/SIBLING_DEPENDENT
verdict is only *half* a finding — the "trigger" must be run alone. Here that turned 28 "graph-opt
SCALE" verdicts into one **operator** bug. The ZEROED dropped-store (§1–7) survived the same check
(the view op is correct alone; only co-aliasing drops it) — so it stays a genuine graph-opt bug.
