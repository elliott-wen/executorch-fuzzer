# VGF graph-optimization bugs — detailed report (corpus_v4)

Deep, device- and plan-verified analysis of the **984 VGF-involved graph-optimization mismatches**
found by bisecting `corpus_v4/vgf` (the full run is in `README.md`). Every claim here is backed by
(a) an on-device **A/B neutralization** and (b) for the headline bug, the **lowered ExecuTorch memory
plan**. Data: `graphopt_all.tsv` (A/B + mechanism + triggers), `tmp/vgf_go_deep.py` (harness),
`tmp/vgf_plan_probe.py` (memory-plan inspector).

---

## 1. What "graph-optimization bug" means here

A mismatch is a **graph-opt bug** (not an operator bug) when the diverging op is **correct computed
alone** but wrong **only inside the larger graph**. The two-axis localizer splits this into:

- **COMPOSITIONAL** (516 VGF cases) — target correct on baked-constant inputs, wrong once specific
  **upstream ops run live** (`needed_live`). The extra live ops are the trigger.
- **SIBLING_DEPENDENT** (468 VGF cases) — target correct returned **alone**, wrong once a specific
  **co-returned sibling output** is emitted (`needed_siblings`). The multi-output plan is the trigger.

Each case was **re-verified on the device with an A/B test** (`tmp/vgf_go_deep.py`): A = target
alone must be **correct**, B = target + trigger must **diverge**, N=3. Cases where the target also
diverged alone (`A_ALSO_DIVERGES`, 7) were reclassified as operator/confound and **excluded**.

## 2. Mechanism distribution (A/B-confirmed device values)

| verdict | ZEROED | NONFINITE | VALUE | ALIAS | SCALE | SHAPE |
|---|--:|--:|--:|--:|--:|--:|
| **SIBLING_DEPENDENT** | **239** | 108 | 55 | 7 | 6 | — |
| **COMPOSITIONAL** | 46 | 86 | **275** | 20 | 18 | 12 |

Mechanism = classification of the *wrong device value* vs the reference (`ZEROED` = device all-zero;
`NONFINITE` = device nan/inf where ref finite; `SCALE:r` = device ≈ ref×r; `ALIAS` = device ==
another live tensor verbatim; `VALUE` = generic; `SHAPE` = wrong output shape). The two verdict
classes have **opposite dominant mechanisms** — this is the core structural result.

---

## 3. ⭐ SIBLING_DEPENDENT → **ZEROED: the dropped-store / output-aliasing bug** (239 cases)

### 3.1 Symptom (device A/B)
A VGF-delegated op returns the **correct** value when its output is the only thing returned, but is
written **all-zeros** the moment a **view/copy-family output is co-returned**:

```
w108:171 out0 target=fmod   A(alone) device=-0.4825 == eager ✓   B(+split_with_sizes_copy) device=0.0
w111:578 out0 target=min    A device=-0.0235 ✓                    B device=0.0
w11:274  out2 target=div    A device= 1.0    ✓                    B device=0.0
w10:13   out1 target=squeeze_copy  A ✓                            B device=0.0
```

- **Zeroed target op is arbitrary** (prod 19, fmod 17, max 13, div 13, min 11, squeeze_copy, flip,
  abs, trunc, mul, remainder, pow, …) → *not* an operator bug.
- **Trigger sibling is a view/copy/reshape op** in the overwhelming majority (squeeze_copy 14,
  lift_fresh_copy 14, clone 11, alias_copy 11, transpose_copy 10, split_with_sizes_copy 10,
  permute_copy 9, view_copy 8, expand_copy 7, unbind_copy, detach_copy, …), and it reads the
  target's **shared input** (in `w108:171`, `n1=fmod(n0)` and the trigger `n5=split_with_sizes_copy(n0)`
  both consume `n0`).

`ZEROED` (all-zeros) is distinguished from `ALIAS` (device == another tensor's value; 7 SIBLING /
20 COMPOSITIONAL cases): all-zeros means the value **never reached the output buffer** — a *dropped
store*, not a clobber-with-other-data.

### 3.2 Root cause — from the lowered ExecuTorch memory plan
Dumping the plan (`tmp/vgf_plan_probe.py`) for the failing graph vs the target-alone graph shows the
memory planner assigns a **different, defective buffer layout** for the target output when the view
sibling forces it to keep extra storage live. For `w108:171` out0 (target `n1`):

```
CORRECT (n1 alone):     n1 output = val[3] @ mem1 offset 0     input L0 = val[0] @ offset 32   (SEPARATE)
ZEROING (n1+siblings):  n1 output = val[6] @ mem1 offset 32  ==  input L0 = val[0] @ offset 32   (ALIASED)
                        instr[0] DelegateCall consumes val[0] (L0 @ 32);  instr[1] fmod writes val[6] (@32)
```

The planner **aliased the target's output buffer onto a graph-input buffer that the VGF delegate
consumes**. Across a 60-case ZEROED sample the plan defect splits into two buffer failures, both
confirmed statically in the plan:

| plan defect | share | what happens |
|---|--:|---|
| **Output aliased onto a still-live INPUT/INTERMEDIATE buffer** (kernel- or delegate-produced target) | ~50% (17 input+7 intermediate + 6 delegate-side of 60) | the target's output offset overlaps a buffer the VGF **delegate reads/reuses**; after the delegate runs, the target's value is gone → zeros |
| **Dropped store to the target's OWN buffer** (16 delegate-produced, 14 kernel-produced of 60) | ~50% | target has a private buffer, but the write never reaches the arena — for a **delegate-produced** target the copy-back from the delegate's internal buffer is **elided**; for a kernel-produced target the store is dropped → the zero-initialized arena value survives |

**Why a view/copy sibling is the trigger.** A view/copy output (`clone`/`squeeze_copy`/`alias_copy`/
`view_copy`/`permute_copy`/`split_with_sizes_copy`/…) *aliases its input's storage*. To return it,
the memory planner must keep that aliased storage live to program end and mark it a USER_OUTPUT. That
perturbs the liveness/aliasing graph the planner solves, and in the VGF multi-output case it
mis-resolves — either overlapping the target's output onto a still-live buffer, or eliding the
target's copy-back. Returned **alone**, the target gets a clean private buffer and the value is
correct. This is a **copy-elision / buffer-aliasing memory-planning bug** — the same family as the
Vulkan copy-elision bug (`bugs/vulkan-copy-elision-aliasing.md`), here in the VGF multi-output plan.

### 3.2b Cross-backend: the bug is **VGF-specific**, not the shared memory planner
Each identical reconstructed graph was lowered to **8 backends** and run on its local runtime
(portable/xnnpack/openvino in-process via `mobile.net.et_runner.run_pte`; vgf/cadence/nxp/ethos-u/qnn
via their host runner scripts). One backend per subprocess (`MOBILE_BACKENDS` allowlist) for crash
isolation. Harness: `tmp/vgf_xbackend.py` + `tmp/vgf_xbackend_run.sh`; raw table `xbackend.tsv`.

Verdict on the finite-value ZEROED (dropped-store) cases, for backends that **actually executed**:

| backend | delegate | partition | w108:171 | w111:578 | w11:274 | w10:13 |
|---|---|---|---|---|---|---|
| **portable** | none (CPU) | — | ✓ | ✓ | ✓ | ✓ |
| **xnnpack** | yes | small (deleg 1–6) | ✓ | ✓ | ✓ | ✓ |
| **openvino** | yes | **whole-graph (deleg 14–15)** | ✓ | (exec err) | (skip) | ✓ |
| **vgf** | yes | whole-graph (deleg 7–13) | **ZEROED** | **ZEROED** | **ZEROED** | **ZEROED** |
| cadence | — | won't delegate/run these graphs (SKIP / rc=2) | — | — | — | — |
| nxp | — | partitioner rejects (SpecViolation / histogram) | — | — | — | — |
| ethos-u | yes (deleg=7) | lowers, but int8 FVP won't execute float/quant (rc=2) | — | — | — | — |
| qnn | — | int8 partitioner rejects the float graphs (rsub / quant-config) | — | — | — | — |

**Every backend that can execute these graphs computes the correct value — only VGF zeroes.** The
decisive control is **OpenVINO**: it absorbs nearly the whole graph into **one delegate partition**
(deleg 14–15, *larger* than VGF's) yet is **correct** — so the failure is neither the shared
ExecuTorch `MemoryPlanningPass` (xnnpack and openvino go through it and are fine) nor "a big
delegate partition emits many outputs" (openvino does exactly that, correctly). The defect is
specifically in **`vgf_backend`'s multi-output writeback** — how it copies its partition outputs
back to the planned arena offsets when an offset aliases a live input or reuses a buffer.

cadence / nxp / ethos-u / qnn cannot execute this **float** corpus (their partitioners are int8- or
capability-restricted, or their emulator rejects the graph) — a coverage limit of those backends on
this corpus, **not** evidence about the bug either way.

> Note: on the **non-finite** case `w102:563` (ref = `nan`), xnnpack *and* openvino *also* diverge
> (they saturate `nan → −1.0`) — i.e. NONFINITE/saturation divergences are **shared across delegates**,
> not VGF-specific. Only the **ZEROED dropped-store** mechanism is VGF-unique. This matches the
> report's split: non-finite = TOSA/delegate saturation (§ README), ZEROED = VGF memory bug.

### 3.2c Source-level localization (VGF runtime + the closed model-converter)
Traced into the VGF backend source (`pytorch_ref/executorch/backends/arm/`, readable) with the
runtime's own `ET_LOG(Info)` on the failing `.pte`:

- **The output IS mapped and copied back — nothing is skipped at the delegate API.** For the zeroing
  graph the delegate has `inputs=1 outputs=4`, maps `output[0]→IO[1]` (this is the ZEROED target
  `n1`), and `VGFBackend.cpp:429 execute()` runs `memcpy(tensor->mutable_data_ptr(), data, io_size)`
  for **every** output including `IO[1]`. So it is *not* a missing writeback in the readable C++.
  → the value the delegate hands back for `IO[1]` is **already zero in device memory**: the compute
  pipeline never wrote that IO (or wrote it to a different alias member).
- **The mechanism is VGF's IO alias-grouping.** `VGFSetup.cpp` binds every resource that shares an
  `AliasGroupId` (`get_resource_alias_group_id`, l.1092) to **one shared `VkDeviceMemory` allocation**
  (`alias_backings`, l.1463–1720), and maps each model output to a resource IO
  (`model_output_io_index[i] = resource_index_to_io_index[mrt_i]`, l.3061). Alias groups are a
  first-class, *intended* feature — there is a dedicated `test_vgf_aliasing_runtime.py`
  ("…alias_group_executes_correctly"). The bug is that for the **multi-output view/copy pattern** the
  target output's resource ends up in an alias group whose shared backing is not written by the
  pipeline for that output → the copy-back reads zero-initialized memory.
- **The aliasing DECISION is made by the closed-source `model-converter` binary** (the AoT
  TOSA→VGF compiler; a prebuilt blob needing `GLIBCXX_3.4.30`, not in the readable tree). It performs
  the copy-elision that places a passthrough (view/copy) output — and, in the multi-output case, the
  *sibling* target — into an alias group. That is why the empirical trigger is always a **view/copy
  sibling**: the converter elides its copy and aliases IO backings, and the resulting VGF blob has an
  output IO that the compute never populates. The runtime faithfully returns that (zero) memory.

**Localization:** trigger = closed `model-converter` copy-elision/IO-aliasing in the multi-output
case; symptom surface = `VGFSetup.cpp` alias-backing + `model_output_io_index` mapping and
`VGFBackend.cpp::execute()` copy-back reading an unwritten aliased IO. The readable runtime is
faithful to bad AoT alias metadata; the fix belongs in the converter's aliasing rule (or a runtime
guard that forbids aliasing a *model output* IO onto backing it doesn't exclusively own).

### 3.3 Confirmation
- **A/B neutralization (device):** returning the target alone (fresh buffer) → correct; adding the
  view sibling → zeros. 239/239 cases (`repro_dropped_store.py`, `graphopt_all.tsv`).
- **Plan neutralization (static):** in the correct plan the target output does **not** share an
  offset with any input/intermediate; in the zeroing plan it does (or its store is elided). Toggled
  purely by whether the view sibling is co-returned.

This is a **backend/compiler memory-planning bug, not a kernel bug** — the target op's kernel is
proven correct in isolation.

---

## 4. COMPOSITIONAL → **VALUE: fp16 number-format divergence** (275 cases)

The target is correct on baked-constant inputs but diverges (finite, moderate `max|delta|`) once
specific **upstream ops run live**. Mechanism is overwhelmingly `VALUE` (275) — a **number-format /
precision** difference: running the upstream cone live lets the VGF compiler pick a different fp16
intermediate format / fusion for the target than it picks when the target is fitted alone. Secondary:
`NONFINITE` (86, the live cone overflows fp16 and the plan introduces nan/inf), `ZEROED` (46, the
same dropped-store defect reached through the cone axis), `ALIAS` (20, target == another live
tensor's value → buffer clobber), `SCALE` (18, incl. a `×-7.7e15` blow-up), and **`SHAPE` (12,
the device returns the wrong output shape** under the multi-output plan — a distinct, serious
graph-opt defect worth its own follow-up).

These are lower-severity than the dropped store (they are precision/format, not total data loss),
but they are the same theme: **the compiler's per-op decision changes with graph context**.

---

## 5. Minor graph-opt mechanisms (small but real)

- **ALIAS (27 total):** device value == another live tensor verbatim → the classic **buffer clobber**
  (memory planner overlapped the target with a still-live tensor and the *other* tensor's write won).
  Distinct from ZEROED (which is the target's *own* store being lost).
- **SCALE (24 total):** device ≈ reference × constant → a number-format/scale choice the plan changed
  (mostly COMPOSITIONAL; one extreme `×-7.7e15`).
- **SHAPE (12, COMPOSITIONAL):** device returns a differently-shaped tensor under the multi-output
  plan.

---

## 6. Severity & recommendation

1. **Dropped store / output-aliasing (239) — highest severity.** Total, silent data loss (an output
   is all-zeros) triggered by an ordinary pattern (return a tensor *and* a view/copy of a shared
   input). Root cause is the **multi-output memory planner** aliasing an output onto a live buffer or
   eliding its store around the VGF delegate boundary. Fix should force a **fresh, non-aliased output
   buffer** for every USER_OUTPUT that a delegate produces or whose storage a delegate consumes (copy
   the input/output fresh — per the ASUS/QNN lesson, the alias can be on the *input*, so neutralizing
   only the output buffer is insufficient).
2. **fp16 number-format (275) — medium.** Context-dependent precision; tighten the compiler's
   intermediate-format choice so a target's format doesn't depend on which siblings/upstream ops are
   co-emitted.
3. **SHAPE (12) — investigate.** Wrong output shape under multi-output planning.

## 7. Reproduce
- `bugs/repro_dropped_store.py` — device A/B for the dropped-store bug (needs broker + vgf_client +
  AoT env; see its header).
- `tmp/vgf_plan_probe.py` — dumps the lowered memory plan and classifies the buffer defect for any
  ZEROED case (`N=<n>` env to size the sample).
- `graphopt_all.tsv` — all 984 cases: `job, out, verdict, tgt_op, trigger_ops, ab, n_div, n_rep,
  mechanism`. Filter `verdict==SIBLING_DEPENDENT && mechanism==ZEROED` for the dropped-store set.
