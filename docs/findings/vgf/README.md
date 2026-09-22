# VGF (Arm TOSA→Vulkan) — corpus_v4 fuzzing analysis

Ran the `analysis.md` workflow on the Arm **VGF** delegate over the multi-op corpus
`corpus_v4/vgf` (**28,878 graphs**, up to 8 nodes each), executed locally on Mesa **lavapipe**
(CPU-software Vulkan) + the ML SDK **emulation layer** via `vgf_client/` (see
[[vgf-emulator-client]]). Every verdict below is **device-verified**; graph-opt cases carry a
device **A/B neutralization**, operator bugs an **N=5 determinism gate**.

## Headline

| outcome | count | % |
|---|--:|--:|
| OK | 16,221 | 56.2% |
| **MISMATCH** | **2,190** | 7.6% |
| SKIP (runtime reject) | 10,358 | 35.9% |
| CRASH (native abort) | 109 | 0.38% |
| TIMEOUT | 0 | 0 |

**Every one of the 2,190 MISMATCH graphs was bisected to a per-job verdict — coverage gate
2,190 / 2,190, zero un-bisected.** (`findings/vgf/bisect_results.tsv`; sweep = 96 sharded
delta-debug processes on the one broker, resumable, poison-capped.)

### The interesting result: **graph-optimization bugs dominate, and VGF's signature mechanism is a DROPPED STORE**

Mismatch verdicts (per-job, two-axis localization):

| verdict | count | axis |
|---|--:|---|
| **COMPOSITIONAL** (graph-opt along the path) | 600 | graph-opt |
| **SIBLING_DEPENDENT** (graph-opt, multi-output) | 475 | graph-opt |
| OPERATOR | 487 | operator (→ 345 VGF, 142 portable-confound) |
| CANT_ISOLATE (target output won't lower alone) | 540 | — |
| POISON (native-crashed ≥6× during localization) | 70 | (crash confound) |
| INCONCLUSIVE | 18 | flaky |

**Graph-opt (1,075 cases, 984 VGF-involved) is 2–3× the operator population** — the opposite of a
single-op corpus. Deep A/B analysis (`findings/vgf/graphopt_all.tsv`,
`tmp/vgf_go_deep.py` — each case re-run: target ALONE must be correct, target+trigger must diverge,
N=3) gives each graph-opt class a **distinct, named mechanism**:

| verdict | dominant mechanism | count (A/B-confirmed) | 2nd | 3rd |
|---|---|--:|---|---|
| **SIBLING_DEPENDENT** | **ZEROED — dropped store** | **239** | NONFINITE 108 | VALUE 55 |
| **COMPOSITIONAL** | **VALUE — fp16 number-format** | **275** | NONFINITE 86 | ZEROED 46 |

(+ small tails: ALIAS 27, SCALE 24, SHAPE 12; ~76 DEV:TIMEOUT hang-prone candidates uncharacterized.)

## Mismatch bugs

> 📄 **Full mechanism analysis with lowered-memory-plan evidence: [`GRAPHOPT_REPORT.md`](GRAPHOPT_REPORT.md).**

### ⭐ Graph-optimization · SIBLING_DEPENDENT · **ZEROED (dropped store)** — the headline bug
`bugs/repro_dropped_store.py` · **239 device-verified cases**

A VGF-delegated op computes the **correct** value when its output is returned **alone**, but is
written **all-zeros** on the device the instant a **view/copy-family output is co-returned**. The
kernel is fine; the **multi-output memory planner drops the target's store** (copy-elision /
buffer-aliasing), leaving the output buffer at its zero-initialized value. Device A/B:

```
w108:171 out[0] target=fmod   A(alone)=-0.4825 ✓   B(+split_with_sizes_copy sibling)=0.0   (eager -0.4825)
w111:578 out[0] target=min    A(alone)=-0.0235 ✓   B(+view siblings)=0.0
w11:274  out[2] target=div    A(alone)= 1.0    ✓   B(+view siblings)=0.0
w10:13   out[1] target=squeeze_copy A=−0.482,0.628 ✓  B(+constant_pad_nd)=0.0,0.0
```

- **The zeroed target op is arbitrary** (top: `prod` 19, `fmod` 17, `max` 13, `div` 13, `min` 11,
  `squeeze_copy`, `flip`, `abs`, `trunc`, `mul`, …) — it is *not* an op bug.
- **The trigger sibling is overwhelmingly a view/copy/reshape op** (top: `squeeze_copy` 14,
  `lift_fresh_copy` 14, `clone` 11, `alias_copy` 11, `transpose_copy` 10, `split_with_sizes_copy` 10,
  `permute_copy` 9, `view_copy` 8, `expand_copy` 7, `unbind_copy`, `detach_copy`, …). Co-returning an
  output that **aliases/views another buffer** is what mis-plans the target's store.

This is the VGF analog of the Vulkan copy-elision/aliasing bug (`bugs/vulkan-copy-elision-aliasing.md`)
— a memory-planning bug in the multi-output plan, **not** any single kernel. Mechanism derived from
*this* backend's device values (all-zeros == dropped store), not assumed from QNN (aliasing) or
Ethos-U (requant-scale).

### Graph-optimization · COMPOSITIONAL · **VALUE (fp16 number-format)** — 275 cases
The target is correct computed alone on baked constants, but diverges (finite, moderate `max|delta|`)
once specific **upstream ops run live** — the compiler picks a different fp16 number-format /
fusion for the target in the larger cone. A minority go **NONFINITE** (86, overflow the plan
introduced) or **ZEROED** (46), plus a few **SCALE** (incl. one `×-7.7e15` blow-up), **ALIAS** (20,
device value == another live tensor's — buffer clobber), and **SHAPE** (12, device returns the wrong
output shape under the multi-output plan).

### Operator bugs (VGF-delegate) — 21 ops confirmed, N=5 deterministic
`bugs/repro_operator_bugs.py` · gate: `findings/vgf/opgate.tsv`

Isolated strictly alone (op on baked-correct constants, `delegated_ops≥1`) and re-run **5×**. VGF
float divergences are cleanly **bimodal** (stable or gone; `INTERMITTENT=0`). Confirmed (5/5):

| op | eager → device (isolated) | nature |
|---|---|---|
| `native_group_norm` | `-1.58,0.74,-0.13,0.97` → `-1.0,1.0,-1.0,1.0` | normalization clamped to ±1 |
| `cumsum` | `1,2` (int) → `15360,16384` | int cumsum returns fp16 bit-garbage |
| `sum` | `1` → `0` | reduction drops the value |
| `floor_divide` | `-1.0` → `-0.0` | sign / rounding |
| `log1p` | `nan` → `-29.5` | wrong domain handling (finite for x<-1) |
| `native_layer_norm`, `asinh`, `_log_softmax`, `prod`, `pow`, `sin`, `remainder`, + 9 singletons (`abs`,`cos`,`fill`,`mean`,`round`,`select_scatter`,`tan`,`var`,`bitwise_xor`) | | 5/5 |

**Ruled out (NOT VGF operator bugs):** `bitwise_left_shift`, `fmod`, `_native_batch_norm_legit`,
`bitwise_right_shift`, `_upsample_bilinear2d_aa`, `maximum`, `var_mean` — these OPERATOR verdicts
run on the **portable fallback** (`delegated_ops=0`), so the divergence is a portable/reference
issue, not VGF (142 mismatch rows total). **Demoted as flaky** (OPERATOR once, 0/5 on re-run,
near-tolerance): `acos`, `addmm`, `div`, `logit`, `sinh`, `_softmax`.

### Non-finite mismatches — mostly saturation semantics, not sink-op bugs
Of the 750 non-finite mismatches, most are **VGF/TOSA saturation** where PyTorch **propagates**
nan/inf: the device produces a *finite* value where the eager reference is nan/inf (e.g.
`ref_nonfinite=3, dev_nonfinite=0`, leaf inputs finite → overflow saturated on device). Filtered as
propagation/saturation per analysis.md, not attributed to the sink op.

## Crash bugs — **0 VGF, 109 portable-fallback** (ruled out)
`findings/vgf/crash_triage.tsv`. All 109 native aborts (`rc=134` SIGABRT) reproduce, and **every one
has `delegated_ops=0`** — VGF delegates nothing in them. They are a **portable-kernel crash cluster**
(98 localize to a single output's cone, 11 multi-output), enriched in `narrow_copy` (30) and
`unfold_copy` (29) *in specific compositions* (both run fine as one-op graphs). Not a VGF-delegate
bug — matches the prior single-op VGF run (0 VGF crashes). This is an ExecuTorch **portable/core**
coverage issue, filed here only as a ruled-out cluster.

## Skip bugs — runtime coverage gaps (10,358; 9,967 runtime-reject + 391 feeder-decode)
`findings/vgf/skip_triage.tsv` (250-sample of the runtime rejects, all reproduce, real reasons from
`run.log`):

| class | share | meaning |
|---|--:|---|
| **VGF-delegate execution failure** (`deleg≥1`, "Execution failed") | 90% (226/250) | the emulation layer can't dispatch the delegated TOSA blob — a VGF **runtime** coverage gap (the `.pte` lowers but won't run) |
| **Missing portable operator** (`deleg=0`) | 10% (24/250) | no CPU kernel — `aten::linear.out` (20), `aten::_fft_r2c.out` (8) (same `_fft_r2c` gap Ethos-U hit), `Dimension out of range` asserts |

The 391 feeder-decode SKIPs are non-tensor outputs (not a backend bug).

## Artifacts
- `bisect_results.tsv` — the coverage-gate table (one verdict row per mismatch job).
- `graphopt_all.tsv` — 984 graph-opt cases, A/B outcome + mechanism + trigger ops.
- `opgate.tsv` — operator N=5 determinism gate.
- `crash_triage.tsv`, `skip_triage.tsv` — crash/skip triage.
- `per_op_tables.txt` — per-op enrichment for every finding class.
- `bugs/` — runnable repros (`repro_dropped_store.py`, `repro_operator_bugs.py`) + `INDEX.md`.
- Tooling (reusable): `tmp/vgf_bisect.py` (two-axis sharded localizer), `tmp/vgf_go_deep.py`
  (graph-opt A/B+mechanism), `tmp/vgf_opgate.py`, `tmp/vgf_crash_triage.py`, `tmp/vgf_skip_triage.py`.

## Reproduce the environment
Broker + workers (device seam): `python -m mobile broker --job-port 15774 --client-port 15775
--ctrl-port 15776` then N× `python vgf_client/vgf_client.py --client-port 15775 --ctrl-port 15776`.
Re-lowering sub-graphs (AoT) needs `model-converter` on PATH + `MODEL_CONVERTER_LIB_DIR` pointing at a
libstdc++ with `GLIBCXX_3.4.30` (`/opt/nvidia/nsight-systems/2024.6.2/host-linux-x64`, cached in
`tmp/vgf_libdir.txt`) — otherwise `build_job` reports `unknown backend 'vgf'`.
