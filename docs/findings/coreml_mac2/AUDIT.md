# AUDIT — CoreML `corpus_v4` graph-optimization claims

Independent, skeptical re-check of `README.md`, `GRAPHOPT_REPORT.md`, `bugs/graphopt_dropped_store.md`,
`bugs/graphopt_scale.md`, `bugs/operator_var_correction.md`.

Everything below is labelled **VERIFIED** (I ran it on this Linux host, or recomputed it from the
archived files), **INFERRED** (follows from verified facts but not directly measured), or
**UNVERIFIABLE NOW** (needs the macOS CoreML worker).

Audit scripts + raw outputs: [`audit/`](audit/) (copied from the scratchpad; `.py` are the probes,
`.txt` their raw results: `sig_*.txt` = AOT signature scan, `xb_*.txt` = cross-backend control,
`planscan.txt` = memory-plan overlap scan).

---

## 0. What could NOT be run, and why

**No macOS CoreML worker is connected.** `ss -ltnp` shows nothing on 15554/15556 and no broker or
worker process exists. Therefore **every device-side claim in the artifact is unverifiable now**:

- the A/B device values in `GRAPHOPT_REPORT.md` §1, §2, §4, §5, §6 and `bugs/graphopt_dropped_store.md`
- the VGF comparison table in §8 (its CoreML column)
- the CoreML column of the `var.correction` table in `bugs/operator_var_correction.md`
- the `SCALE` / `ZEROED` / `WRONG-VALUE` labels themselves, which are functions of device values

`bugs/repro_alias_dropstore.py` and `bugs/repro_graphopt.py` both require `BROKER_PORT`; they exit
without a worker. I did not fabricate any device number.

**What I could run instead.** `coremltools 9.0` **is** installed here (only `libcoremlpython`, the
macOS *runtime*, is missing), so `get_backend("coreml")` returns a working backend and the whole
**ahead-of-time** half of the pipeline — partitioning, CoreML/MIL conversion, the delegate blob, the
ExecuTorch memory plan — is fully reproducible on Linux. That turned out to be far more productive
than expected: two of the artifact's central claims can be *proved at AOT* without a Mac, and one
named "device-verified" case can be *refuted* on this host.

---

## 1. Number reconciliation — 204 vs 503 vs 50 vs 48 (VERIFIED, recomputed)

### 1a. The 503 and the 204 are consistent with each other

`graphopt_mechanisms.tsv` has exactly **503** data rows, of which exactly **204** carry mechanism
`ZEROED`. So "**204 of 503**" (`GRAPHOPT_REPORT.md` §Scope; `bugs/graphopt_dropped_store.md`
Prevalence; README mechanism table) **reconciles exactly with the archived file**. ✔

The rest of that table also reconciles: WRONG-VALUE 125, REF-NONFINITE 85, NEAR-TOL 19, NONFINITE 17,
NO-REPLAY 1 — and `204+125+85+19+17+1 = 451`, leaving **52 `SCALE:r` rows** (see 1c).
"Filtering 104 non-device cases leaves ~399" checks out (503 − 85 − 19 = 399). ✔

### 1b. The 50 is a *different, now-unrecoverable* population

README's headline paragraph says "**50 of 109** bisected graph-opt cases", and `GRAPHOPT_REPORT.md`
§4 says "**Of the 50 corpus ZEROED cases**, 1 is the pure two-alias case; 49 are …". Two facts:

1. There is no 109-row file in the artifact. `characterize_graphopt.py` **overwrites**
   `graphopt_mechanisms.tsv` on every run, so the 109-row / 50-ZEROED snapshot from the partial
   bisection has been destroyed. The 50-case population is **not reconstructible from the artifact.**
2. README therefore contradicts itself: its headline says 50/109 and its own mechanism table two
   sections later says 204/503.

**These are different populations.** The **204** is the full-corpus count. The **50** is an interim
snapshot taken while the bisection was still running. Everything §4 and
`bugs/graphopt_dropped_store.md` say about *composition* was computed on the 50, then published under
a 204 headline. Recomputing the same statistics on the real 204 (script `tally.py`) shows the
transplant does not hold:

| published (from the 50-snapshot) | recomputed on the actual 204 |
|---|---|
| target-op table summing to **28** cases (`lift_fresh_copy` 10, `squeeze_copy` 7, `alias_copy` 3, `view_copy` 3, `t_copy` 3, `detach_copy` 2) | `lift_fresh_copy` **29**, `alias_copy` **17**, `clone` **17**, `squeeze_copy.dims` **17**, `t_copy` **11**, `squeeze_copy.dim` **9**, `unfold_copy` 5, `slice_copy` 4, `detach_copy` 3, `transpose_copy` 2, `unsqueeze_copy` 2 … |
| "target ops are **overwhelmingly** no-op/view/identity" | view/copy/identity family ≈ **116 / 204 = 57%**. The other ~43% are real compute ops: `tan.out` 4, `logical_and` 4+2, `prod.out` 4, `logical_not.out` 4, `prod.default` 3, `fmod` 5, `erf` 2, `asinh` 2, `any.dims` 2, `max` 4, `roll` 2, `mean` 2, `sub`, `div`, `log1p`, `repeat`, `select_scatter`, `index`, `bitwise_left_shift` 2 … |
| trigger sibling "`var.correction` **×15**" | in the 204, `var_mean.correction` appears as a required sibling **4×** and `var.correction` **0×**. Top triggers are `_to_copy` 8, `remainder.Scalar_out` 8, `any.dims_out` 6, `broadcast_to` 5, `atan2` 5 — a long flat tail, as claimed, but the ×15 figure is not in this population. |
| "1 pure two-alias / 49 boundary-view" (§4) | on the 204 the AOT signature split is **14 two-alias (DUP) / 87 boundary-view (PASSTHRU) / 103 neither** (§4 below). The *ordering* survives (boundary-view dominates); the *ratio* does not. |

**Verdict on 1b:** "204 of 503" is defensible as a *count*. The *characterisation* attached to it
(target-op table, "overwhelmingly view/identity", "var.correction ×15", "1 vs 49") is a 50-case
statistic presented as a 204-case statistic and must be recomputed or re-scoped.

### 1c. The SCALE count is wrong in every place it appears

| published | where | actual |
|---|---|---|
| "SCALE:r **~26**" | README mechanism table | **52** |
| "**48** `SCALE:r` cases" | `GRAPHOPT_REPORT.md` §9 | **52** |
| "**48** corpus cases classified `SCALE:r`" | `bugs/graphopt_scale.md` | **52** |
| "28 of **48**" | `bugs/operator_var_correction.md` | denominator wrong |

`mechanism_dist.txt` (the archived stdout of the classifier) also sums to 52 SCALE rows, so 48 and
~26 are not from any archived run. The `~26` in README is presumably "SCALE rows minus the
`SCALE:0` mislabels" but is never defined.

### 1d. The verdict table has drifted from the current TSVs

The 6 `bisect_results.*.tsv` files contain 2,494 data rows for **1,528** unique `(job_id, out_idx)`
keys (the resume logic appends, so 966 rows are re-attempts — a naive `wc -l`/`grep -c` over these
files over-counts by ~63%). Deduplicated (last row wins):

| verdict | README | actual now |
|---|--:|--:|
| CONE_LOCAL_CANT_SPLIT | 656 | **662** |
| OPERATOR | 328 | **331** |
| SIBLING_DEPENDENT | 308 | **310** |
| COMPOSITIONAL | 195 | **196** |
| INCONCLUSIVE | 3 | **4** |
| **real verdicts** | **1,490 (93.1%)** | **1,503 (93.9%)** |
| SIB + COMP ("503") | 503 | **506** |

So the recovery pass added rows after `characterize_graphopt.py` ran. The published numbers are a
consistent *earlier* snapshot, not errors — but README claims the recovery pass was still improving
coverage, so the table should be regenerated before submission. `1,601` mismatches and the
feed line `OK 5,321 · MISMATCH 1,601 · CRASH 325 · SKIP 3,250 · TIMEOUT 21` reconcile exactly with
`skiplog.tsv`. ✔

One further inconsistency: README's Coverage section says **SKIP 3,250 / CRASH 325** while its "light
tally" section and `skips.md` say **SKIP 2,247 / CRASH 221** ("so far"). The former are correct; the
latter are stale interim counts that read as final.

---

## 2. Cross-backend control — the highest-value check (VERIFIED)

This is the claim that makes the bug CoreML-specific rather than an ExecuTorch memory-planning bug.
**It largely holds, but not universally, and one of the report's named cases is refuted.**

### 2a. Reduced repro probes: 10/10 correct on portable AND xnnpack

I rebuilt every construct in `GRAPHOPT_REPORT.md` §1/§2/§4 from scratch (`probes.py`) and ran each
in-process under `MOBILE_BACKENDS=portable` and `MOBILE_BACKENDS=xnnpack`:

| probe | portable | xnnpack |
|---|---|---|
| `lift_fresh_copy(n2)` alone | OK | OK |
| `(lift_fresh_copy(n2), alias_copy(n2))` | **OK** | **OK** |
| `(n2, alias_copy(n2))` | OK | OK |
| `(alias(L0), alias(L0))` | OK | OK |
| `(lift_fresh, alias, view_copy)` ×3 aliases | OK | OK |
| `(alias(n2), alias(n3))` control | OK | OK |
| `+0.0` neutralized | OK | OK |
| §4 unsupported source (`div.Scalar_mode`) | OK | OK |
| §4 supported source control (`mul.Scalar`) | OK | OK |
| §1 `w1:8` reduced (`cosh→div→(max, lift_fresh_copy)`) | OK | OK |

`user_pos = [0,1]` (or `[0,1,2]`) in every case, in order — so the CUDA-run alignment trap
(`user_pos` misalignment masquerading as a dropped store) is genuinely **not** present here. ✔
**The report's §1 cross-backend table is reproducible on this host.**

### 2b. Corpus scale: 184/204 confirmed correct on portable — but 3 are refuted

I ran the **full graph** of all 204 ZEROED corpus cases and compared *the same output index*
(`xback.py`):

| | portable | xnnpack |
|---|--:|--:|
| OK (target output correct) | **184** | 175 |
| MISMATCH | 2 | 2 |
| MISMATCH + all-zero (i.e. also ZEROED) | **1** | **1** |
| RUN_EXC (no host kernel — untestable) | 17 | 17 |
| BUILD:SKIP | 0 | 9 |

**Three cases diverge identically on portable and xnnpack, deterministically 3/3 on each**
(`recheck.py`):

| job | out | dtype | eager | portable & xnnpack |
|---|--:|---|---|---|
| **w101:421** | 3 | int32, numel 1 | `[3]` | `[0]` ← all-zero |
| w124:557 | 2 | float32, numel 288 | `[0,0,1,1,1,1,…]` | `[0,0,0,0,0,0,…]` |
| w65:544 | 1 | int64, numel 48 | `-3,-3,-3,0,-3,-1…` | differs later, max\|Δ\|=4 |

**`w101:421` out[3] is listed in `GRAPHOPT_REPORT.md` §5 as a device-verified CoreML instance with
`[3.0] → [0.0]`.** Portable and xnnpack reproduce `[3] → [0]` exactly. Its graph is
`n2 = cumsum.out(n0, 0, out=n1)` — an `out=` written into `n1`, which is itself live downstream —
i.e. a backend-independent `out=`-aliasing hazard in the host pipeline. **This case is not
CoreML-specific and must be removed from the §5 table.** (It is also `NOT-DELEGATED`: see §4.)

**Scope limit (negative result):** 17 of 204 cases raise in the portable kernel and 9 more fail to
build on xnnpack, so **for 17 cases the cross-backend control cannot be run at all on this host** —
neither confirmed nor refuted.

**Bottom line:** 184/204 (90.2%) positively confirmed CoreML-specific at the corpus level, 3/204
(1.5%) refuted, 17/204 (8.3%) untestable. That is a genuinely strong control — stronger than the
report's own evidence, which only tested a hand-built reduced graph — but it is **not "the identical
lowered subgraph is correct on portable and xnnpack"** without exception, as §1 states.

---

## 3. The ExecuTorch-plan check — the claim SURVIVES, with better evidence (VERIFIED)

`GRAPHOPT_REPORT.md` §3 claims the alias-to-one-buffer decision happens **inside the compiled CoreML
model**, because at the ExecuTorch level both aliases are clean `USER_OUTPUT`s. §3 shows only the
*graph*, not the memory plan, so it does not actually establish this. I checked the plan.

**Graph structure (VERIFIED, `graph_dump.py`)** — reproduces §3 verbatim for the two-alias probe:
one `executorch_call_delegate`, two `getitem`s, two `OutputKind.USER_OUTPUT` specs, and the same
structure survives `to_executorch()`. For the §4 probe, `aten::div.Scalar_mode` stays outside as a
portable op and **both** `max` and `lift_fresh_copy` land inside one delegate.

**Memory plan (VERIFIED, `plan_dump.py`)** — deserialised `.pte`, `plan.outputs` →
`Tensor.allocation_info`:

| probe | out[0] alloc | out[1] alloc | overlap? |
|---|---|---|---|
| `(lift_fresh_copy(n2), alias_copy(n2))` | mem 1 @ **16** (16 B) | mem 1 @ **0** (16 B) | **none** |
| `(n2, alias_copy(n2))` | mem 1 @ 16 | mem 1 @ 0 | none |
| `(alias(L0), alias(L0))` | mem 1 @ 16 | mem 1 @ 0 | none |
| ×3 aliases | @32 / @16 / @0 | | none |
| §4 unsupported source | @16 / @0 | | none |
| `+0.0` neutralized | @16 / @0 | | none |

Every returned tensor gets its own byte range, and no two `USER_OUTPUT`s share a value index.
Scaled to the whole population (`planscan.py`): **204 of 204 ZEROED corpus cases lower successfully here and every one is `PLAN_DISJOINT`** —
no two `USER_OUTPUT` byte ranges overlap in any of them (I compared real byte extents, `memory_id` +
`memory_offset_low` + `numel × itemsize`, not just offsets). Zero counterexamples.

So **the "inside the compiled CoreML model" claim is correct** and the fault attribution stands.
(The task flagged that `Verifier.verify_storage_reuse` in `exir/memory_planning.py` is unreachable
under the default `MemoryPlanningAlgorithmSuite` and so would not catch a bad plan — true, but moot
here: the plan is clean by direct measurement, not by trusting the verifier.)

### 3a. NEW, host-only: the actual mechanism, visible AOT in the delegate header

The ExecuTorch→CoreML delegate blob begins with a JSON header naming the CoreML model's input and
output variables. Dumping it (`mil2.py`) makes the bug visible **without a Mac**:

| probe | delegate header | predicted |
|---|---|---|
| `(lift_fresh_copy(n2), alias_copy(n2))` | `in ["x"]`, `out ["aten_mul_scalar", "aten_mul_scalar"]` | **same name twice** → one buffer, one write |
| `(alias(n2), alias(n3))` control | `out ["aten_mul_scalar", "aten_add_scalar"]` | distinct → correct |
| `+0.0` neutralized | `out ["aten_mul_scalar", "aten_add_scalar"]` | distinct → correct |
| §4 `div.Scalar_mode` (unsupported) source | `in ["aten_div_scalar_mode"]`, `out ["aten_max_default", "aten_div_scalar_mode"]` | **an output that IS the delegate's own input** |
| §4 `mul.Scalar` (supported) source control | `in ["x"]`, `out ["aten_max_default","aten_mul_scalar"]` | neither → correct |

Both aliasing ops are elided to their producing variable, and the ExecuTorch CoreML delegate then
binds two distinct output buffers to **one** CoreML model output name — exactly the "binds them to
one buffer, writes it once" mechanism, now with **AOT evidence instead of device inference**. The
`+0.0` neutralization works because it forces a distinct MIL variable. This makes §2/§4 two
*mechanically distinct* signatures:

- **DUP** — the same name appears twice in `outputNames` (§2, two returned aliases of one tensor)
- **PASSTHRU** — an `outputName` is also an `inputName`, i.e. the CoreML model must return one of
  its own inputs verbatim (§4, view of a portable-fallback tensor)

§4's wording ("the aliased source isn't a materialized delegate output") is directionally right but
imprecise; "the delegate is asked to return its own input" is the checkable statement.

### 3b. NEW: the signature is strongly enriched in ZEROED — quantified (VERIFIED, `sig_scan.py`/`okctl.py`)

I lowered every graph-opt case here and classified the *specific* target output:

| population | n | DUP | PASSTHRU | signature-positive | rate |
|---|--:|--:|--:|--:|--:|
| **ZEROED** | 204 | 14 | 87 | **101** | **49.5%** |
| WRONG-VALUE | 125 | 5 | 7 | 12 | 9.6% |
| SCALE:* | 52 | 2 | 3 | 5 | 9.6% |
| NEAR-TOL | 19 | 0 | 0 | 0 | 0% |
| device-**OK** control (150 random OK jobs, 584 user outputs) | 584 | **0** | 8 | 8 | **1.4%** |

**35× enrichment over device-OK outputs, 5× over other mismatch mechanisms.** No device-OK output
carries the DUP signature at all. This is independent, Mac-free corroboration that the mechanism is
real and is what drives the ZEROED class. **This is the strongest single result in this audit and the
report does not contain it.**

Two negative results in the same data:

1. **~half the ZEROED class is unexplained by the stated mechanism.** 78/204 are `NEITHER` and
   **25/204 (12.3%) are `NOT-DELEGATED`** — the op producing the zeroed output is a *portable kernel*
   running on the host inside the CoreML `.pte`, not CoreML code at all. Two of the eight cases in
   the §5 "confirmed instances" table are in this group: **w101:421** (`aten_fmod_scalar`; already
   refuted in §2b) and **w103:475** (labelled "`alias`-family" in §5 but actually
   `aten_prod_default`, a reduction, not delegated). For these the sentence "CoreML lowering must
   materialize a distinct output buffer" cannot be the fix.
2. **The signature is not sufficient.** 8 device-OK jobs carry a PASSTHRU output and are correct, and
   1 signature-positive case (w65:544, PASSTHRU) fails on portable too. So a further condition exists
   that only a Mac can pin down.

Signatures for the §5 table (VERIFIED):

| job | out | §5 label | AOT signature |
|---|--:|---|---|
| w1:8 | 3 | `lift_fresh_copy(n2=div)` | PASSTHRU (`aten_div_scalar_mode`) |
| w1:781 | 1 | `lift_fresh_copy` | PASSTHRU (`…clone_dim_order_default`) |
| w1:489 | 5 | `lift_fresh_copy` | **DUP** (`aten_tan_default`) |
| w101:421 | 3 | (copy-family) | **NOT-DELEGATED** — and **refuted**, fails on portable |
| w103:475 | 1 | `alias`-family | **NOT-DELEGATED**, actually `aten_prod_default` |
| w104:767 | 4 | `alias_copy` | PASSTHRU (`aten_detach_copy_default`) |
| w111:17 | 3 | `view_copy` | NEITHER |
| w118:301 | 0 | `squeeze_copy.dims` | PASSTHRU (`aten_fmod_scalar`) |

5 of 8 corroborated at AOT; 1 refuted; 2 carry no signature (one of them mislabelled).

---

## 4. Classifier defects (VERIFIED, `characterize_graphopt.py` + `dtype_scan.py` + `scale_audit.py`)

The two precedents named in the brief — `findings/vgf/shape_probe/README.md` (SHAPE 0/12 real, from a
bare element-count test) and `findings_v2/ethos_u_fvp/REQUANT_SCALE_REVIEW.md` (SCALE 1-of-3 real,
vacuous on a constant reference) — both apply here, unevenly.

### 4a. ZEROED is a genuine POSITIVE test — it does **not** have the SHAPE defect

```python
if all((isinstance(v,(int,float)) and abs(v)<1e-12) for v in et) and any(abs(v)>1e-9 for v in finite(eager)):
    return "ZEROED"
```
Device must be **all** ≈0 **and** the reference must have a real non-zero value, and the
reference-non-finite filter runs first. This is a two-sided positive test with a
degenerate-reference guard — structurally sound, and categorically better than the VGF SHAPE test
(`numel(dev) != numel(ref)`). **No vacuity defect.** Two real but narrower weaknesses:

- **No dtype guard.** 67 of 204 ZEROED outputs are non-float: 58 integer (`int64` 25, `int32` 17,
  `uint8` 9, `int16` 7) and **9 boolean**. For a bool output "all zeros" just means *all False*, an
  utterly ordinary wrong answer; 11 of the ZEROED *target ops* are logical ops
  (`logical_and` ×6, `logical_not` ×4, `logical_or`). Calling those an "uninitialized buffer" is
  unsupported. For integers, exact 0 is also a common legitimate result (`fmod`, `remainder`,
  `prod`, `bitwise_*` all appear in the target set).
- **numel==1 carries no information.** 39 of 204 ZEROED outputs have exactly one element, and 85 of
  204 have a fully constant reference. "The whole buffer read back zero" is only evidence of a
  dropped store when the buffer has several independently-computed elements. For the 39 one-element
  cases, ZEROED is indistinguishable from "wrong by one value".

Combining: **~118 of 204** (float, numel > 1) are cases where the ZEROED label carries its intended
meaning; ~86 need re-wording as "wrong value that happens to be zero". Cross-cut against §3b, the
subset that is *both* multi-element float *and* signature-positive is the publication-grade core.

### 4b. SCALE has **exactly** the ethos-u defect, worse

```python
ratios=[e/g for g,e in zip(eager,et) if abs(g)>1e-6 ...]
if all(abs(r-r0)<0.05*max(1,abs(r0)) for r in ratios) and abs(r0-1.0)>0.08:
    return f"SCALE:{r0:.3g}"
```
The magnitude filter is present and correct, but the consistency test has the same two blind spots
the ethos-u review documented — it passes trivially on a **single** surviving ratio and on a
**constant** reference — *plus* a third: `0.05*max(1,|r0|)` degenerates to an **absolute** 0.05
tolerance whenever `|r0| < 1`, so for `r0 = 0.5` any ratio in `[0.45, 0.55]` (a 20 % spread) counts
as "consistent".

Measured on all 52 SCALE rows (`scale_audit.py`, reference recomputed from `build_job(...).eager`):

| | n | share |
|---|--:|--:|
| **numel == 1** (one ratio → test vacuous) | **38** | 73 % |
| multi-element but **constant reference** (test vacuous) | 4 | 8 % |
| **VACUOUS total** | **42** | **81 %** |
| test has real discriminating power | **10** | 19 % |

**81 % of the SCALE labels rest on a test that cannot fail.** Only 10 of 52 are real measurements of
a proportional relationship. And 4 rows are labelled `SCALE:0`, which is not a scale at all — three
of them are plain logical/value inversions (`w100:723` int16 `[1,0,0,0]→[0,1,1,1]`; `w108:526` bool;
`w44:585` int32). README half-admits this ("`SCALE:0` mislabels") but the category is still published.

### 4c. The "28 of 48 are `var.correction`" re-triage (VERIFIED, partly)

| claim | measured |
|---|---|
| denominator 48 | **52** |
| "28/48 are the var bug" | **29 of 52** SCALE rows are in graphs containing `var/var_mean/std.correction`. Exactly **28** of those are also in the VACUOUS set — so the published 28 is almost certainly "SCALE rows whose graph contains `var.correction`", i.e. **attribution by graph containment, not per-case verification.** |
| "Factor is **exactly** (N−1)/N (×0.5 at N=2, ×0.75 at N=4)" | Only **20 of the 29** var-graph rows have a ratio matching `(N−1)/N` within 1.2 %. **9 do not**, including `1.17`, `1.20`, `1.25` — ratios **> 1**, which a (N−1)/N *scale* cannot produce — plus `−1.18`, `0.485`, `0.565`, `0.632`, `0.821`, and the `SCALE:0` bool inversion `w100:723`. Where the downstream consumer is nonlinear (`sinh`, `asinh`, `tan`, `log`) the observed ratio is *not* (N−1)/N even when var is the true cause, so "exactly (N−1)/N" is the wrong supporting statement. |
| the (N−1)/N signature is specific to var graphs | **No.** **5 non-var** rows also match `(N−1)/N` within 1.2 % (0.75, 0.807, 0.857, 0.901) — a direct demonstration of 4b's vacuity: on a one-element constant reference some `(N−1)/N` always fits by chance. |
| "no other mechanism hides in the 48" | **Not established.** Of the 10 SCALE rows with real discriminating power, only **1** is a var graph (and its "scale" is 0). The other 9 are unrelated and un-triaged: `w107:89` ×−1 (384 elems), `w35:295` ×0.5 (64 elems, `−0.0` reference), `w47:599` ×2 (int64), `w56:665` ×2.3e18 (`bitwise_left_shift`, an overflow/int64-sentinel confound of exactly the kind the ethos-u review flagged), `w63:511` ×0 (`max_pool2d_with_indices_backward`), `w79:529` ×−0.167, `w117:292` ×−0.00787, `w108:526`/`w44:585` bool/int inversions. `bugs/graphopt_scale.md` calls these "a heterogeneous tail … candidates, not filed", which is honest — but they are precisely the rows where the classifier had power, and they are the ones left un-triaged. |

### 4d. Non-defect: the REF-NONFINITE filter is correct

`if any(nf(v) for v in eager): return "REF-NONFINITE (filter)"` runs before every other test, so the
85 reference-degenerate rows are excluded from all device categories. ✔

---

## 5. The `var.correction` OPERATOR bug — CONFIRMED at AOT, no Mac needed (VERIFIED)

This is the audit's second new result. `bugs/operator_var_correction.md` claims CoreML computes the
**biased** variance (÷N) for `correction=None`. I confirmed it from the **MIL the converter emits on
this Linux host** (`varmil2.py`), and pinned the source line.

```
correction=None  →  reduce_mean(mean) · sub · square · reduce_mean            ← ÷N, BIASED
correction=0     →  … reduce_mean · mul(1.0)                                  ← ÷N   (correct)
correction=1     →  … reduce_mean · mul(1.3333334)  = ×N/(N−1) at N=4         ← ÷(N−1) (correct)
```

Root cause, `coremltools/converters/mil/frontend/torch/ops.py`, `_var()` (~line 7479):

```python
variance = mb.reduce_mean(x=x_demeaned_square, axes=axes, keep_dims=keep_dims)  # biased
if unbiased or correction is not None:      # <-- both are None when correction is omitted
    ...                                     #     so the debias multiply is SKIPPED
```

ATen's `var.correction` default is `correction=1` (unbiased ÷(N−1)); the converter's gate treats
`None` as "no correction requested" and returns the biased result. Off by exactly (N−1)/N.
`std.correction` routes through the same `_var(..., correction=correction)` (line 7742) and inherits
the defect. `var_mean.correction` has no handler in this `ops.py` and is decomposed earlier.

**This is publication-grade and is stronger than the artifact's own claim**: it is an *upstream
coremltools converter* bug with a named file/function/gate, reproducible on Linux with no Apple
hardware, not a runtime or ExecuTorch bug. The write-up's "CoreML's `var`/`std` lowering must honor
`correction=None` as correction=1" is the right fix; the attribution should name coremltools.
(The `portable`/`coreml` *numeric* table in that file remains device-dependent and unverifiable now.)

---

## 6. Determinism bookkeeping (VERIFIED)

**No repetition count is recorded anywhere.** `bisect_results.*.tsv` columns are
`job_id / out_idx / verdict / cone_size / required / detail`; `graphopt_mechanisms.tsv` is
`job_id / out / verdict / required / mechanism / sample`. Neither has an N or reps column, and no log
records one. `bisect_driver._run()` executes each candidate graph **once** per predicate evaluation,
and `characterize_graphopt.replay()` replays each job **once**. So:

- **Mechanism labels (ZEROED / SCALE / …) are single-observation, N = 1.** `GRAPHOPT_REPORT.md`
  §Scope's "determinism-stable on replay" is **not supported by anything in the artifact.**
- **Verdicts have incidental N ≥ 2 evidence, and it is good.** Because the resume logic re-attempts
  rows, **693** `(job, out)` keys received ≥ 2 independent real verdicts, and only **2 disagreed**
  (`w68:271`, `w102:100`) — **691/693 = 99.7 % reproducible**. Another 18 agreed on the verdict while
  reporting the required-sibling set in a different order (set-ordering, not instability). This is
  real determinism evidence for the *localization* axis and the report should cite it.
- The 19 `NEAR-TOL (gate N>=5)` rows are correctly flagged as awaiting a determinism gate that was
  never run. The VGF precedent is directly relevant: there, **4 of 12** SHAPE cases evaporated on
  re-run and they were exactly the rows already flagged `n_div=1/3`.

---

## 7. Verdicts

### Publication-defensible as stated

1. **`var.correction` biased-variance operator bug** (§5). Now *stronger* than published: AOT-confirmed
   on Linux, root-caused to a named gate in the coremltools torch frontend. Re-attribute to
   coremltools; add `std.correction`.
2. **"204 of 503" as a count of rows in `graphopt_mechanisms.tsv`** (§1a).
3. **The ExecuTorch memory plan does not alias the two outputs** (§3) — verified by direct
   `allocation_info` dump on the probes and at population scale, so "the decision happens inside the
   compiled CoreML model" stands.
4. **The ZEROED classifier is a real positive test**, not a residual (§4a) — unlike the VGF SHAPE
   precedent.
5. **The reduced §1/§2/§4 constructs are correct on portable and xnnpack** (§2a), including clean
   `user_pos`, so the CUDA alignment trap is genuinely excluded.
6. **The dropped-store mechanism itself**, now with AOT corroboration the report lacks: duplicated /
   input-aliased `outputNames` in the delegate header, 49.5 % of ZEROED vs 1.4 % of device-OK
   outputs (§3a–3b).

### Needs re-wording before submission

7. **Separate the 204 from the 50.** README's "50 of 109" headline must go, and every composition
   statistic derived from the 50 (target-op table, "overwhelmingly view/identity", "`var.correction`
   ×15", "1 pure two-alias vs 49 boundary-view") must be recomputed on the 204 or explicitly scoped
   to the interim snapshot. On the 204: view/identity is **57 %**, not "overwhelming", and 12 % of
   the class is not delegated at all.
8. **Fix the SCALE denominator everywhere: 52, not 48 or ~26.** Restate "28 of 48 are var" as
   "29 of 52 SCALE rows occur in graphs containing `var/std.correction`", label it attribution by
   containment, and **drop "the factor is exactly (N−1)/N"** — only 20 of 29 match, 3 have ratios > 1,
   and 5 non-var rows match too.
9. **Report the SCALE category as 81 % vacuous** (38 one-element + 4 constant-reference of 52), citing
   the ethos-u precedent, or do not report a SCALE category at all. The 9 non-var rows where the test
   *did* have power are the un-triaged ones.
10. **Remove `w101:421` from the §5 device-verified table** — it fails identically on portable and
    xnnpack (3/3 each). Re-check `w103:475` (mislabelled "alias-family"; actually a non-delegated
    `prod`). Restate §1 as "184 of 204 corpus cases confirmed correct on portable" rather than a
    universal claim, and disclose the 17 untestable cases.
11. **Qualify ZEROED by dtype and numel.** 9 bool + 58 integer outputs and 39 one-element outputs
    cannot support an "uninitialized buffer" reading. Quote the multi-element float subset (~118) as
    the class, or the signature-positive multi-element float subset as the core.
12. **Regenerate the verdict table** (506/1,503/93.9 %, not 503/1,490/93.1 %) and reconcile
    SKIP 3,250 vs 2,247 / CRASH 325 vs 221.
13. **State N = 1 explicitly** for mechanism labels; cite the 691/693 re-attempt agreement as the
    determinism evidence for verdicts; keep NEAR-TOL gated.

### Needs a Mac to settle

14. Every device value in §1, §2, §4, §5, §6 and the §8 VGF/CoreML comparison. In particular the §2
    minimal-probe table — the *ordering* of which output zeroes, and the `+0.0` fix — is the crux and
    is currently unverifiable.
15. Why **103 of 204** ZEROED cases carry **no** AOT signature, and why 8 device-OK jobs carry the
    PASSTHRU signature yet are correct. The signature is necessary-ish but not sufficient; the
    missing condition is device-side.
16. Whether the 25 `NOT-DELEGATED` ZEROED cases are a *different* bug (portable kernel + CoreML
    delegate interaction) rather than the CoreML output-binding bug they are currently filed as.
17. `characterize_graphopt.py` overwrites `graphopt_mechanisms.tsv` in place. Make it write a
    timestamped file — the 109-row snapshot that §4 and the target-op tables were computed from is
    permanently lost, which is why those numbers cannot be audited at all.
