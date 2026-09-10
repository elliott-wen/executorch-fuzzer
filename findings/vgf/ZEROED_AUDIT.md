# ZEROED "dropped store" — independent audit of the headline VGF claim

Audit target: `GRAPHOPT_REPORT.md` §3 + `README.md` — **289 ZEROED** graph-optimization cases
(`graphopt_all.tsv` col 9 == `ZEROED`: 243 `SIBLING_DEPENDENT`, 46 `COMPOSITIONAL`; the report
publishes **239** by dropping 4 `A_ALSO_DIVERGES` SIBLING rows).

Everything below was **run**, not read. Scripts and raw results: `findings/vgf/zeroed_audit/`
(`ab_all289.jsonl`, `plan_all289.jsonl`, `comp_ancestors.jsonl`, `per_case_table.md`,
`xbackend_sibling60.txt`). Every claim is marked **[V]** verified by execution or **[I]** inferred.

## Verdict up front

| claim | status |
|---|---|
| The A/B device symptom (target correct alone → all-zeros with the trigger) | **[V] HOLDS — 289/289, zero flakes** |
| "Device output is genuinely all-zeros, eager genuinely non-zero" | **[V] HOLDS — 285/285 measured** |
| The **239 SIBLING_DEPENDENT** figure | **[V] DEFENSIBLE** (in fact 243 reproduce; VGF-specific on 44/44 cross-checked) |
| The **46 COMPOSITIONAL** additions to the 289 | **[V] MOSTLY SPURIOUS — ≥41/46 are upstream single-op bugs, 4 are not even VGF; ≤5 survive** |
| §3.2 root cause: "~50% output aliased onto a still-live INPUT/INTERMEDIATE" | **[V] FALSE — 0/60 and 0/285 once liveness is checked** |
| §3.2 "the trigger sibling is a view/copy op in the overwhelming majority" | **[V] FALSE — 115/239 have any view/copy trigger; and view/copy is neither necessary nor sufficient** |
| §3.3 "Plan neutralization (static)" | **[V] NOT SUPPORTED by the cited probe** (it never lowers the A side; its overlap test has no liveness check) |
| A minimal trigger exists | **[V] FOUND — 2 lines, deterministic** |

**Publishable count: ~243 (SIBLING_DEPENDENT only), not 289.** The symptom is real, deterministic
and VGF-specific; the *mechanism story* in §3.2 must be replaced.

---

## 0. Environment (verified live before any measurement)

```bash
export PYTHONPATH=/data/jwen929
export CUDA_VISIBLE_DEVICES=
export MODEL_CONVERTER_LIB_DIR=/opt/nvidia/nsight-systems/2024.6.2/host-linux-x64
export PATH=/data/jwen929/mobile/.venv/bin:$PATH
python -c "from mobile.gen.export.backends import get_backend; print(get_backend('vgf'))"
# -> <mobile.gen.export.backends.vgf.VgfBackend object at 0x...>      [V] backend live
```
All device runs go straight to `vgf_client/vgf_runner.sh` (lavapipe + ML SDK emulation layer),
outputs parsed from `out_<i>.bin`. No broker, no worker fleet.

---

## 1. Reproduction rate — the WHOLE population, not a sample

The brief asked for ≥25 cases. Lowering turned out to cost ~1.5 s, so **all 289 rows** were re-run.

```bash
cd findings/vgf/zeroed_audit
awk -F'\t' '$9=="ZEROED"{print $1, $2}' ../graphopt_all.tsv > cases_all289.txt   # 289
for i in $(seq 0 11); do ZA_TMP=$PWD/fr$i python audit.py \
    --cases cases_all289.txt --out full.s$i.jsonl --shard $i --nshards 12 & done; wait
```
`audit.py` rebuilds the A and B graph **from scratch on every rep** (so AoT non-determinism is
in scope), N=3 per side, and replays the harness's own comparison path
(`cmp.select(user_pos)` → trailing-trim → `cmp._cmp`).

* A side: `SIBLING_DEPENDENT` → `output_set_localizer` `bos([target])`;
  `COMPOSITIONAL` → `cone_localizer` `build_src([])`.
* B side: `bos([target]+needed_siblings)` / `build_src(needed_live)`.

**[V] Result — 289/289 reproduce exactly as claimed:**

| outcome | SIBLING | COMP | total |
|---|--:|--:|--:|
| **A correct 3/3 AND B all-zeros 3/3** (the claim) | **243** | **46** | **289** |
| flaky (A or B mixed) | 0 | 0 | 0 |
| A also diverges (⇒ operator bug) | 0 | 0 | 0 |
| build/run failure now | 0 | 0 | 0 |

Notes:
* [V] The 4 rows the original run recorded as `A_ALSO_DIVERGES` and **excluded**
  (`w101:730/1 tanh`, `w121:756/1 flip`, `w127:226/0 repeat`, `w17:556/1 masked_scatter`)
  all give A = `OK,OK,OK` and B = `ZEROED×3` here. `vgf_go_deep.py:analyze_sibling` gates A on a
  **single** rep (`aliveA = diverges_at(...)` once), so that gate is noisy — here it erred toward
  *under*-counting.
* [V] 4 rows initially failed inside *my* parser on a 0-byte output file (empty-tensor sibling)
  — a bug in my `to_tensors`, not in the artifact. Fixed and re-run
  (`w93:520/0 div`, `w14:533/1 max`, `w85:22/3 tan`, `w21:490/1 prod`): all `O/O/O` → `Z/Z/Z`.

Per-case table (all 289 rows, with A/B verdicts, plan class and the ancestor probe):
`findings/vgf/zeroed_audit/per_case_table.md`.

This is the **opposite** of the SHAPE category. The SHAPE collapse was driven by rows recorded at
`n_div=1/3`; **[V] every one of the 289 ZEROED rows is recorded `n_div=3/3`**
(`awk -F'\t' '$9=="ZEROED"{print $7"/"$8}' graphopt_all.tsv | sort -u` → `3/3` only), so the
low-determinism over-representation that killed SHAPE is simply absent here.

## 2. Is it really ZEROED? — yes

`diag.py` dumps, per case, the raw `out_*.bin` byte lengths, an all-zero test on every output file,
the eager |max| per output, `user_pos`, the plan output count and the plan geometry.

```bash
for i in $(seq 0 11); do ZA_TMP=$PWD/fd$i python diag.py \
    --cases cases_all289.txt --out fdiag.s$i.jsonl --shard $i --nshards 12 & done; wait
```

**[V]**
* **285/285** measured cases: the device buffer selected for the target is **bit-exactly all zeros**
  (`|max| < 1e-6`), and the eager reference for that same output has `|max| > 1e-3`. Not
  "small-but-non-zero". (4 cases hold an empty sibling tensor and were skipped by this probe only;
  they are confirmed ZEROED by `audit.py` above.)
* **226/239** SIBLING cases: the target is the **only** wrong output — every co-returned sibling is
  `OK`. 13 have a second wrong output too. Clean, isolated data loss.
* Confound 4c (output misalignment / the `net/compare.py::select` silent fallback) **does not fire**:
  283/285 have `n_eager == n_plan_out == n_out_files` and `user_pos == [0..n-1]`. The 2 exceptions
  (`w47:758/1`, `w49:772/0`) have a non-identity but fully **in-range** `user_pos`
  (`[1,2]`, `[2,3]`), so `select` takes the intended branch — no fallback.
* An extra check: for each case, *every* device output file was reinterpreted as the target's
  dtype/shape and compared. In 9 of 24 spot-checked SIBLING cases another file *does* match the
  target's eager value — but in all 9 the two eager outputs are numerically identical
  (e.g. `w55:603` `relu`/`relu`, `w77:21` `atanh`/`atanh`), i.e. a duplicate-value coincidence,
  **not** an off-by-one read. Important as a clue (see §5), harmless as a confound.

## 3. Plan-offset evidence — §3.2's table is an artifact

### 3a. The report's own probe reproduces its number
```bash
N=60 python tmp/vgf_plan_probe.py
#  17 OUT_ALIASES_INPUT/kernel      15 OWN_BUFFER/prod=delegate    14 OWN_BUFFER/prod=kernel
#   7 OUT_ALIASES_INTERMEDIATE/kernel  4 OUT_ALIASES_INTERMEDIATE/delegate  3 OUT_ALIASES_INPUT/delegate
```
**[V]** 31/60 = 52% "aliased" — matches §3.2's "~50%". So the number is real; the *interpretation*
is what fails.

### 3b. `tmp/vgf_plan_probe.py` has three defects (code read + measured)
1. **No liveness test.** It calls any byte-range overlap in the arena an alias. Memory planning
   *reuses dead buffers by design*; overlap alone is not a defect.
2. **It never lowers the A side.** It lowers `bos(ret)` — the full-return graph — so it cannot
   perform the "plan neutralization" §3.3 claims (correct plan vs zeroing plan). It also indexes
   `plan.outputs` by position in `ret`, ignoring `user_pos`.
3. **Wrong element-size table** (`SZ={0:1,1:2,...,8:8}`): ScalarType 1 = `Char` is 1 byte, not 2;
   8 = `ComplexHalf` is 4, not 8. Inflated ranges inflate the overlap count.

### 3c. Re-measured with liveness, on the report's exact 60 cases and exact graph
`fullret.py` reproduces `vgf_plan_probe.py`'s geometry (same `bos(ret)` graph, same 60 rows, its
buggy size table for the "report" column) and adds def/last-use intervals from the instruction
stream (graph inputs live from −1, graph outputs live to program end).

```bash
ZA_TMP=$PWD/runfr python fullret.py
```
**[V] Result — every "alias" is legal reuse of a DEAD buffer:**

| report category | n | corrected (liveness-aware) |
|---|--:|---|
| `OUT_ALIASES_INPUT/kernel` | 17 | `OWN_BUFFER (dead reuse)` |
| `OUT_ALIASES_INTERMEDIATE/kernel` | 7 | `OWN_BUFFER (dead reuse)` |
| `OUT_ALIASES_INTERMEDIATE/delegate` | 4 | `OWN_BUFFER (dead reuse)` |
| `OUT_ALIASES_INPUT/delegate` | 3 | `OWN_BUFFER (dead reuse)` |
| `OWN_BUFFER/*` | 29 | `OWN_BUFFER (no overlap)` |

**0/60** targets overlap a simultaneously-live buffer.

### 3d. The same measurement on the real A/B pair, all 285 cases (`plananal.py`)
```bash
python plananal.py 'fdiag.s*.jsonl'
```
**[V] Measured split — the answer to "what is the actual plan defect":**

| target output's buffer, B-side plan | n | share |
|---|--:|--:|
| overlaps a **simultaneously-live** graph INPUT | 0 | **0%** |
| overlaps a **simultaneously-live** intermediate | 0 | **0%** |
| own buffer, legal reuse of a dead buffer | 228 | 80% |
| own buffer, no overlap at all | 57 | 20% |

So §3.2's table ("~50% aliased onto a still-live buffer / ~50% dropped store") is **0% / 100%**.
There is **no memory-planner aliasing defect** in this data.

**[V] What the plan *does* show — the correct static signature.** Splitting each `DelegateCall`'s
args into inputs (already-defined values) and outputs (first definitions):

| | B side | A side |
|---|--:|--:|
| plan contains a `DelegateCall` with **≥2 outputs** | **243 / 285** | 17 / 285 |
| by verdict | SIBLING 237/239, COMP 6/46 | — |
| A→B transition `(no multi-out → multi-out)` | **226 / 285** | |

For SIBLING_DEPENDENT this is a clean neutralization: adding the sibling turns a single-output VGF
partition into a multi-output one. It is also the pointer to the real mechanism (§5).
For COMPOSITIONAL it is *not* present (only 6/46) — a first sign that the COMP bucket is a
different animal (§4d).

## 4. Confounds

### 4a. "Trigger is a view/copy-family op" — **FALSE as stated**
`GRAPHOPT_REPORT.md` §3.1 lists `squeeze_copy 14, lift_fresh_copy 14, clone 11, alias_copy 11, …`
and concludes "in the overwhelming majority". Those are just the most *frequent individual names*
in a long tail. Counting whole rows (view/copy set = squeeze/lift_fresh/clone/alias/transpose/
split/permute/view/expand/unbind/detach/t/narrow/slice/unsqueeze/select/_to/broadcast/reshape/
flatten/unfold/chunk `_copy` family):

**[V]** SIBLING_DEPENDENT ZEROED, n=239: **115 have *any* view/copy trigger; 124 have none.**
COMPOSITIONAL ZEROED, n=46: 15 any, **0 all-view**. Non-view triggers are common and ordinary:
`constant_pad_nd` 10, `abs` 7, `mul` 6, `log10` 6, `prod` 5, `pow` 5, `fmod` 5, `div` 5,
`var_mean` 4, `relu` 3, `amax` 3, `acosh` 3, `sigmoid` 2, `_native_batch_norm_legit` 1 …

**[V]** And the minimal probes (§5) settle it: `relu`+`relu` **zeroes** with no view op anywhere,
while `relu` + `view_copy(L0)` and `relu` + `transpose_copy(n0)` are **correct**. View/copy is
neither necessary nor sufficient. §3.2's "why a view/copy sibling is the trigger" paragraph is
explaining a pattern that is not in the data.

### 4b. Portable fallback (`delegated_ops == 0`) — 4 cases, and they are NOT VGF
**[V]** 4 of 289 have **no `DelegateCall` at all** in the B plan (all COMPOSITIONAL):
`w49:644/0`, `w74:450/3`, `w93:501/0`, `w60:389/1`.
```bash
MOBILE_BACKENDS=vgf,portable,xnnpack ZA_TMP=$PWD/runpc python portcheck.py
# w49:644 0 n3 | vgf(deleg=0)=ZERO  portable(deleg=0)=ZERO  xnnpack(deleg=0)=ZERO
# w74:450 3 n7 | vgf(deleg=0)=ZERO  portable(deleg=0)=ZERO  xnnpack(deleg=0)=ZERO
# w93:501 0 n3 | vgf(deleg=0)=ZERO  portable(deleg=0)=ZERO  xnnpack(deleg=0)=ZERO
# w60:389 1 n5 | vgf(deleg=0)=ZERO  portable(deleg=0)=ZERO  xnnpack(deleg=0)=ZERO
```
All four zero identically on the **portable** and **xnnpack** runtimes. They are ExecuTorch
portable-kernel-vs-ATen divergences with VGF nowhere in the program. **Must be subtracted.**

### 4c. Output misalignment / `select()` silent fallback — **ruled out** (see §2).

### 4d. NEW, and the largest confound: COMPOSITIONAL ZEROED is mostly an *ancestor operator* bug
`cone_localizer`'s A side **bakes the ancestors as constants**, so a `COMPOSITIONAL` verdict never
tests whether the live ancestors themselves compute the right value. If a live ancestor returns
zeros, the target is fed zeros and its output is zero — an ordinary single-op bug wearing a
graph-opt label.

`comp_probe.py` takes each COMP case's B graph and re-emits it returning **one live ancestor
alone**, N=3 on the device:
```bash
awk -F'\t' '$9=="ZEROED" && $3=="COMPOSITIONAL"{print $1,$2}' ../graphopt_all.tsv > comp46.txt
for i in 0 1 2 3 4 5; do ZA_TMP=$PWD/cp$i python comp_probe.py \
    --cases comp46.txt --out comp.s$i.jsonl --shard $i --nshards 6 & done; wait
```
**[V] Result, n=46:**

| | n |
|---|--:|
| **at least one live ancestor is ALREADY wrong/zeroed returned ALONE** | **41** |
| all live ancestors individually correct (candidate genuine graph-opt) | **4** — `w33:310/1`, `w61:595/0`, `w76:777/1`, `w84:104/0` |
| ancestor not runnable alone (`rc=2`), undecidable | 1 — `w3:174/0` |

Broken-ancestor ops (device-vs-eager, returned alone): `fill` **14× ZERO**, `_to_copy` 4 ZERO/3 BAD,
`bitwise_left_shift` 4 BAD, `slice_scatter` 3 ZERO, `prod` 3 ZERO, `floor_divide` 3 ZERO/2 BAD,
`pow`/`view_copy`/`sigmoid`/`trunc`/`broadcast_to` 2 each, plus `remainder`, `cumsum`, `mean`,
`native_group_norm`, `argmin`, `cos`, `neg`, `exp`, `le`, `roll`, `index_select`, `div`, `mul`, …

Independently confirmed by hand: **[V]** `torch.ops.aten.fill.Scalar(L0, 3.5)` as a **single-op**
VGF graph returns **all zeros**, 3/3. `full_like` too. That one operator bug alone accounts for 14
of the 46 "COMPOSITIONAL graph-opt ZEROED" cases.

**The SIBLING_DEPENDENT half is immune to this**: `output_set_localizer` does *no* baking — the A
side keeps the real leaves and the full live computation and only changes which outputs are
returned. A correct A therefore already proves the ancestors are fine. This asymmetry between the
two localizer axes is the single most important methodological finding of this audit.

### 4e. VGF-specificity of the surviving set — **holds, and on far more than 4 cases**
The report's cross-backend control (§3.2b) uses 4 cases. Extended to 60 (`xcheck.py`,
in-process `mobile.net.et_runner.run_pte`):
```bash
MOBILE_BACKENDS=portable,xnnpack python xcheck.py --cases x60.txt --shard $i --nshards 6
```
**[V]** 48 SIBLING rows in that window; 44 executed. **44/44 portable = OK**, **42/42 xnnpack = OK**
(2 xnnpack `BUILD:SKIP`, 4 unrunnable on both). Zero non-VGF reproductions. Only VGF zeroes.

## 5. The minimal trigger — **found**, and it is not what §3.2 says

`minimal.py` / `minimal2.py` / `minimal3.py` hand-build tiny graphs, lower them to VGF, dump the
plan and run N=3.

```bash
ZA_TMP=$PWD/runm  python minimal.py
ZA_TMP=$PWD/runm2 python minimal2.py
ZA_TMP=$PWD/runm3 python minimal3.py
```

### The 2-line repro
```python
import torch
L0 = torch.randn((2, 4), dtype=torch.float32)
def g(L0):
    n0 = torch.ops.aten.relu.default(L0)
    n1 = torch.ops.aten.relu.default(L0)   # syntactically DUPLICATE subexpression
    return (n0, n1)
LEAVES = [L0]
```
**[V] out0 = all zeros, out1 = correct. 3/3 reps. float32 and float16.**
Plan: one `DelegateCall` `args=[0,1,2]` — 1 input, 2 outputs, `deleg=2`.
`out=[(1,mem1,32,32),(2,mem1,0,32)]`, no overlap with anything live.

### Boundary map (all [V], N=3 each)

| graph | result |
|---|---|
| `relu(L0)` alone | ok |
| `relu(L0), abs(L0)` (2 outputs, **distinct** ops) | ok, ok |
| `relu, abs, neg` / 4 distinct outputs | all ok |
| **`relu(L0), relu(L0)`** | **ZERO, ok** |
| `relu, relu, relu` (3 duplicates) | **ZERO, ZERO, ok** — all but the last |
| `relu, relu, neg` | **ZERO**, ok, ok |
| `neg, relu, relu` | ok, **ZERO**, ok — only the duplicate group is hit |
| `add.Scalar(L0,1), add.Scalar(L0,1)` | **ZERO, ok** |
| `sigmoid, sigmoid` | **ZERO, ok** |
| `relu(L0), relu(L1)` (duplicate op, **different** inputs) | ok, ok |
| `relu(L0), abs(L0)` with `L0 > 0` (**same value**, different op) | ok, ok |
| `relu(L0), abs(relu(L0))` (same value, chained) | ok, ok |
| `return (n0, n0)` (literally the same node twice; plan emits one value index twice) | ok, ok |
| `n0=relu(L0); n1=expand_copy(n0,[2,4])` (identity-shaped passthrough) | **ZERO, ok** |
| `n0=relu(L0); n1=view_copy(n0,[4,2])` (different shape) | ok, ok |
| `n0=relu(L0); n1=transpose_copy(n0,0,1)` | ok, ok |
| `n0=relu(L0); n1=slice_copy(n0,0,0,2)` (whole-tensor slice, folded before emit) | ok, ok |
| dup producers whose outputs feed **portable** kernels (`fmod.Scalar`, `min`) | **ZERO, ok** |
| distinct producers whose outputs feed the same portable kernels | ok, ok |
| `n0=remainder(L0,L1); n2=lift_fresh_copy(n0); return (min(n0), log10(n2))` | **ZERO, ok** |
| `relu(L0), clone(L0)` | not runnable (`rc=2`) — separate issue |

### Statement of the trigger
**[V]** A VGF partition that must emit **two or more model outputs carrying the same tensor
produced by duplicate or identity-equivalent computations** returns **all but one of them as
all-zeros**. Multi-output alone is not the trigger (4 distinct outputs are fine); value equality
alone is not the trigger (`relu`/`abs` on positive input is fine); a *single* plan value returned
twice is fine. It is the AoT converter collapsing duplicate producers.

**[V]** The last two rows of the table are exactly the corpus's own paradigm cases reduced by hand:
`w108:171` (`index(L0,[0,0])` → `fmod.Scalar(n0,6)` + `split_with_sizes_copy(n0,[2],0)`; the
delegate emits `n0` twice, once for the portable `fmod`, once as the split) and `w111:578`
(`remainder` → `min(n0)` + `log10(lift_fresh_copy(n0))`). This is why the target op is arbitrary and
why the trigger op looks like a view/copy in the corpus *sometimes* — a no-op copy is just one of
many ways to make the converter emit the same tensor twice.

**[I]** Mechanism: this is consistent with §3.2c's `VGFSetup.cpp` alias-group localization — the
converter binds the duplicated outputs into one `AliasGroupId` / one `VkDeviceMemory` backing and
populates only one of the corresponding output IOs; `VGFBackend.cpp::execute()` then faithfully
`memcpy`s zero-initialized memory for the others. §3.2c is the part of the report that survives.
§3.2's memory-planner story does not.

## 6. Negative results (all of them)

* No case was flaky. No case failed to build or run. No case diverged on the A side. (Contrast:
  4 of 12 SHAPE cases were flaky.)
* No case had a small-but-non-zero device value masquerading as zero.
* No case was mis-selected by `net/compare.py::select` — the silent-fallback hazard that produced
  the SHAPE artifact **did not fire** for ZEROED.
* No case had its output buffer overlapping a simultaneously-live buffer — the hypothesis §3.2
  builds its root cause on is 0/285.
* `value equality without syntactic duplication` does **not** trigger the bug (tested 2 ways).
* `return (n0, n0)` does **not** trigger it — the duplication must survive to the delegate boundary.
* `view_copy` / `transpose_copy` passthroughs do **not** trigger it; `expand_copy` at the same shape
  does. So "any aliasing view op" is too coarse a characterisation.
* xnnpack and portable never reproduce (44/44, 42/42).
* Two probe scripts of my own initially crashed on 0-byte (empty) output tensors; that was my bug,
  fixed, and the 4 affected cases confirm.

## 7. Recommended edits before submission

1. **Reduce the headline to the SIBLING_DEPENDENT set.** 239 as published is defensible and
   conservative; 243 is what actually reproduces (add back the 4 mislabeled `A_ALSO_DIVERGES`
   rows and note that the A gate used one rep). **Do not claim 289.**
2. **Drop the 46 COMPOSITIONAL ZEROED from the graph-opt bug count.** 41/46 are upstream single-op
   device bugs the cone axis cannot see (it bakes the ancestors); 4 are pure portable-kernel
   divergences with no VGF delegate. Re-file the big one — **`aten.fill` returns all zeros on VGF,
   14 cases** — as an operator bug; it is a good finding on its own.
3. **Delete §3.2's plan-defect table and the "~50% aliased onto a live buffer" claim.** Replace the
   static evidence with the measured signature: *a `DelegateCall` with ≥2 outputs appears in 243/285
   B-side plans vs 17/285 A-side plans (226 flip A→B)*.
4. **Replace §3.1's "trigger is a view/copy op" with the measured trigger**: duplicate /
   identity-equivalent producers among a partition's outputs. Lead with the 2-line repro.
5. **Fix or retire `tmp/vgf_plan_probe.py`** (no liveness test, never lowers the A side, wrong
   element-size table for `Char`/`ComplexHalf`).
6. **Fix the localizer methodology note**: `COMPOSITIONAL` verdicts need an
   *ancestors-correct-alone* gate, exactly as `SIBLING_DEPENDENT` gets its target-alone gate.
   Without it the COMP axis silently relabels ancestor operator bugs as graph-opt bugs. This
   probably affects the other COMP buckets in `GRAPHOPT_REPORT.md` §4 (275 VALUE, 86 NONFINITE,
   20 ALIAS, 18 SCALE) — **not audited here, but the same hole applies.**
7. **Fix `net/compare.py::select`** to raise on an out-of-range `user_pos` (per the SHAPE report).
   It did not cause this artifact, but it remains a live hazard.
