# AUDIT — `bugs/graphopt-fill-buffer-aliasing.md` (QNN HTP x86 emulator, `corpus_v2/qnn`)

Skeptical review of the ">=56% of the 2,936 GRAPHOPT verdicts are ONE buffer-aliasing bug" claim.
Every number below was recomputed from the artifact TSVs; **the emulator was brought up and the
claim was re-run on device** (300 fresh cases, 3 repetitions each).

Evidence markers: **[V]** verified on this host · **[I]** inferred · **[U]** unverifiable now.

---

## 0. Verdict per claim (detail follows)

| # | claim | verdict |
|---|---|---|
| 1 | ">=56% of the 2,936 GRAPHOPT verdicts are ONE buffer-aliasing bug" (37% sibling-specific + 19% all-sibling) | **UNSUPPORTED AS STATED.** Arithmetic is clean and the sample is sound, but the 19% sub-bucket is an **op-name heuristic, not a measurement**, and the 37% sub-bucket is **not distinguishable from a no-sibling-effect noise model**. A positively identified buffer-clobber is ~2-3%. Defensible replacement: **35.0% [29.9-40.1] device-confirmed sibling-dependent graph-level defects**, headline signature **ZEROED/dropped-store (39% of those)**. |
| 2 | the constant-output probe method + the three device-verified `fill` examples | **NEEDS RE-WORDING.** The probe is a sound *wrongness* oracle (43/43 held up on re-run) but does **not** discriminate aliasing from mis-lowering. "byte-identical"/"100% match" is **false as written** (the test is a 5% relative tolerance). `w100:66` reproduces and is genuinely byte-exact; `w102:448` is vacuous; `w103:514` matches nothing and should not be cited as mechanism evidence. |
| 3 | `min.unary_out` (`w100:389`) is correct with 3 of 4 siblings, wrong only with the `fill` sibling | **PUBLICATION-DEFENSIBLE — and stronger than stated.** Re-verified 3/3 on device. Two corrections: the device value is `2e-05`, not `0.0`; and the gold-standard *alone* control **is** available for this target (it passes 3/3), so the write-up under-sells its own strongest case. |

---

## 1. Number reconciliation — the arithmetic is clean

### 1.1 Bisection totals **[V]**

`bisect_results.tsv` = 19,923 data rows (header + 19,923 = 19,924 lines). Verdict column:

| verdict | recount | WORKFLOW_RESULTS.md | agree |
|---|--:|--:|:--:|
| OPERATOR | 14,984 | 14,984 | yes |
| **GRAPHOPT** | **2,936** | **2,936** | yes |
| NOREPRO | 1,976 | 1,976 | yes |
| BISECT_CRASH | 15 | } 27 | yes |
| BISECT_HANG | 10 | } | yes |
| LOWERFAIL | 2 | } | yes |
| total | 19,923 | 19,923 | yes |

The triage table (20,618 / 19,923 / 818 / 19,981) and the "19,923 verdict rows == 19,923 mismatch
targets" coverage gate both hold. **No bookkeeping error at the bisection stage.**

### 1.2 The 587-case sweep **[V]**

`tmp/sweep_shards/*.tsv` contains exactly **587 unique (job, out) records**. (Note: the file is
**malformed** — every record is split across two physical lines because `sweep_batch.py` builds
`CASES` with `ln.split("\t")[:2]` and never strips the trailing newline, so `outidx` is `"0\n"`.
Recoverable, but a reproducibility defect worth fixing.)

| sweep class | count | share |
|---|--:|--:|
| GRAPHOPT | 218 | 37.14% |
| OPERATOR | 164 | 27.94% |
| INCONCL | 133 | 22.66% |
| FLAKY | 72 | 12.27% |

Every bucket in the bug file's table reconciles **exactly**:

| bug-file bucket | stated | derivation | recomputed |
|---|--:|---|--:|
| sibling-specific | 37% | 218/587 | 37.14% |
| all-sibling output-aliasing | 19% | **110**/587 | 18.74% |
| genuine-operator candidates | <=9% | (164-110)=54 /587 | 9.20% |
| flaky | ~12% | 72/587 | 12.27% |
| inconclusive | ~23% | 133/587 | 22.66% |
| **sum** | **100%** | | **100.01%** |

And 37.14 + 18.74 = **55.9% ~= the stated 56%**. **The arithmetic is internally consistent and
sums to 100%.** No double-counting, no stage confusion in *this* table.

### 1.3 Was the 587 a random sample? **YES — and this is a genuine strength [V]**

I ranked each swept case by its position among the 2,936 GRAPHOPT rows in `bisect_results.tsv`:
ranks are **exactly** 5, 10, 15, ..., 2935 — every rank divisible by 5, zero exceptions, 587x5=2935.

So the 587 is an **exact 1-in-5 systematic stride**, not "the first 587". Prefix coverage confirms
uniformity (200 of the first 1000, 300 of the first 1500, 400 of the first 2000 — exactly 20% of
every prefix). File order is by job-id string, uncorrelated with target op. **The extrapolation
587 -> 2,936 is statistically sound.** This is better practice than the artifact claims for itself
and should be stated explicitly in the paper (it currently just says "a 587-case sample").

### 1.4 One real bookkeeping error **[V]**

> "`fill.Scalar` is the **#1 GRAPHOPT target (520 cases)**"

`520` is the `fill.Scalar` row of `op_enrichment.txt`, whose own header reads *"MISMATCH attribution
(op producing the diverging out[N])"* — that is the count over **all 19,923 mismatches**, not over
the 2,936 GRAPHOPT verdicts. In that table `fill.Scalar` is **#6**, behind `logit` (791),
`logit.out` (719), `select_scatter` (627), `rsub.Scalar` (565) and `sub.out` (527).

The true GRAPHOPT-only count, from the 1-in-5 sample, is **91 x 5 ~= 455**. `fill.Scalar` *is* the
#1 GRAPHOPT target (91/587 = 15.5%, the largest single op) — so the ranking claim survives, but the
**number is wrong and comes from the wrong stage**. This is precisely the error the Ethos-U review
caught ("21 DRIFT + 5 CLOBBER is the *pinpoint* stage, not a mechanism count").

---

## 2. Method soundness

### 2.1 What `sweep_batch.py` actually measures **[V]**

For each (job, target): pair the target with each other returned output (max 6), build the
cone-reduced 2-output subgraph, run it **once**, and count `nok` / `nmm` / `nsk`. Then:

```
GRAPHOPT = nok>0 and nmm>0 ;  OPERATOR = nmm>=2 and nok==0 ;  FLAKY = nok>0 and nmm==0 ;  else INCONCL
```

Three structural defects:

1. **No ALONE control.** The classifier never establishes "target correct alone" — the actual
   definition of a graph-opt bug. A stricter classifier that *does* run the alone control and adds a
   `CONFOUND` bucket exists in this repo (`tmp/pinpoint_graphopt.py`, written 2 days *after* the bug
   file) but its `BK`/`CO` are hard-coded to **`ethos-u`**; it was never pointed at QNN. **[V]**
2. **One run per pair, on the project's least-deterministic target.** `WORKFLOW_RESULTS.md` itself
   says the QNN emulator has 7 non-deterministic kernels and is "the least deterministic target
   tested". A single observation per pair cannot separate "sibling identity matters" from
   "row-level propensity + per-run noise". This is the same defect the VGF SHAPE review caught
   (rows flagged `n_div=1/3` "were carried into the category anyway") — here it is systemic.
3. **`nsk` conflates four different outcomes**: build-SKIP, device SKIP, device CRASH, and a 50 s
   `TIMEOUT` — all increment `nsk`. Under 32-way parallelism, timeouts silently convert an OK into a
   skip, which flips GRAPHOPT -> OPERATOR/INCONCL.

**Defect 3 is not hypothetical — it corrupted the write-up's own centrepiece [V].** The sweep row for
`w100:389` (`min.unary_out`) is:

```
w100:389   0   INCONCL   nok=0  nmm=1  nsk=3   min.unary_out
```

i.e. *zero* siblings OK. My device re-run (3 reps each) gives **`+n3` OK, `+n6` OK, `+n7` OK,
`+n5` MISMATCH** — the three "skips" are stable OKs. The sweep's per-row data is demonstrably
unreliable on the one case the bug file leans on hardest.

### 2.2 The 37% bucket is not distinguishable from noise **[V]**

Restricting to the 446 rows with >=2 resolved siblings, pooled per-run P(OK) = 576/1247 = 0.462.

**Test A — plain binomial null (no sibling effect, no row heterogeneity):**

| | mixed (=GRAPHOPT) | all-MM (=OPERATOR) | all-OK (=FLAKY) |
|---|--:|--:|--:|
| observed | 218 | 164 | 64 |
| binomial null | 298.5 | 87.4 | 60.1 |

Observed "mixed" is *below* the null, i.e. outcomes are **over-dispersed** — rows behave
consistently. Good: there is real per-row structure. But over-dispersion is exactly what *row-level
heterogeneity* produces, and it is **not** evidence of a sibling-identity effect.

**Test B — beta-binomial null: rows differ in their OK-rate, and *sibling identity carries zero
information*:**

| population | obs mixed | BB-null mixed | unexplained excess |
|---|--:|--:|--:|
| all classifiable (n=446) | 218 | **207.4** | **+10.6 rows (+2.4 pts)** |
| exact-valued targets (n=294) | 116 | 110.4 | +5.6 (+1.9 pts) |
| float/transcendental (n=152) | 102 | 96.1 | +5.9 (+3.9 pts) |

A **two-parameter model with no sibling effect whatsoever reproduces the observed 218/164/64 split
to within ~11 rows.** Under the plain binomial the transcendental subgroup fits *exactly*
(102 observed vs 102.9 expected). **The sweep's "37% sibling-specific" figure carries almost no
discriminating power** — it is what a single-run-per-pair protocol returns from row heterogeneity
plus per-run noise. This is QNN's analogue of the Ethos-U "test passes trivially" defect.

`fill.Scalar` is the reverse case and is very clean: **0 mixed rows out of 66**, per-run OK-rate
**1.7%**. Fill is wrong in essentially *every* context — the signature of an unconditional
operator/lowering defect, not a sibling-dependent one.

### 2.3 The 110 reclassified rows: **circular — assigned by op name [V]**

The bug file concedes the sweep cannot see an all-sibling output-aliasing bug, then asserts
"the `fill`/`copy` cases (**110 of the 164** 'OPERATOR'-bucket rows)" are really aliasing, "proved"
by the clobber probe. Asking what independent evidence assigns those 110:

Op histogram of the 164 OPERATOR-bucket rows:

```
ops whose NAME contains "fill" or "copy"                        97
  (fill.Scalar 65, copy.default 24, masked_fill 3, view_copy 1,
   squeeze_copy 1, slice_copy 1, select_copy 1, lift_fresh_copy 1)
+ roll.default                                                   7
+ select_scatter.default                                         6
                                                            =  110      <-- exactly
```

**110 = 97 + 7 + 6 reproduces the number exactly.** The 110 is a **name-based bucket**
("fill/copy/roll/select_scatter are known-aliasing ops"), not a set of device-probed rows.

Corroborating: the bug file is dated 2026-06-30 08:31 and the only clobber-probe runs that predate
it are the **three** manual `pinpoint_clobber.py` invocations reported in its own tables (plus one
`copy` and one `masked_fill`). The 32-shard batch pinpoint (`tmp/pinpoint/`) is dated 2026-07-02 —
*after* — and targets **ethos-u**. **[V]**

So: **~5 device-probed cases are generalised to 110 by op name, then reported as a measured 19%.**
The reclassification is circular as written. What *is* defensible: "fill/copy targets are wrong with
every sibling, so a sibling-variation test cannot classify them; on device they are provably wrong
(re-run below), but whether that wrongness is aliasing or mis-lowering is undetermined."

### 2.4 Is the clobber probe a POSITIVE test? **Not as described [V]**

`tmp/pinpoint_clobber.py::matches()`:

```python
if af.numel()!=devf.numel():
    if af.numel()>devf.numel() and af.numel()%1==0:
        af=af[:devf.numel()]          # any LARGER tensor is truncated to a prefix
    else: return None
d=(af-devf).abs(); rel=d/(devf.abs()+1e-3)
frac=float((rel<0.05).float().mean()); return frac
```

- The criterion is a **5% relative tolerance**, and "100% match" means *100% of elements within 5%*.
  The write-up's "**byte-for-byte**" / "**byte-identical**" wording is **factually wrong**.
- Every larger candidate is **truncated to a prefix**, inflating the candidate space.
- No informativeness guard: a **1-element** output or an **all-zero** output matches almost anything.
- **The probe cannot fail.** `w103:514` matched nothing and was still assigned to the same bug as
  "an internal/reused scratch or `out=` buffer". Match -> aliasing; no match -> aliasing. No outcome
  refutes the hypothesis. This is a new defect not present in either precedent.

---

## 3. Re-run on device — the emulator WORKS **[V]**

The x86 HTP emulator is fully usable and **needs no broker**: `qnn_client.QnnExecutor` can be driven
in-process. Setup that worked:

```bash
source /data/jwen929/mobile/tmp/qnn_env.sh     # -> android-env.sh: QNN_SDK_ROOT=.../qairt/2.37.0.250724
export MOBILE_BACKENDS=qualcomm CUDA_VISIBLE_DEVICES= PYTHONPATH=/data/jwen929
# runner: pytorch_ref/executorch/build-x86/examples/qualcomm/executor_runner/qnn_executor_runner
```

Cost: ~2-3 s to lower a subgraph, **~0.5-1 s per device run**. Gotcha: without `qnn_env.sh` every
`build_job` returns `SKIP` with a *libQnnSystem.so* error — **indistinguishable from "won't
partition"**, which is itself a hazard for any sweep that treats SKIP as "unlowerable".
`Failed to set powerConfig`/`Failed to get device` on stderr are benign.

### 3.1 The four named jobs (3 reps each; byte-exactness tested by true `torch.equal`)

| job | target | eager | sibling | device (3/3 stable) | byte-exact match |
|---|---|---|---|---|---|
| `w100:66` | `fill.Scalar(n0,7)` | `[7,7]` | `+n4` | `[-0.48242, 1.11035]` | **`L0`, UNIQUE** |
| | | | `+n6` | `[-0.48242, 1.11035]` | **`L0`, UNIQUE** |
| | | | `+n3` | `[nan, 0.0]` | none |
| | | | `+n5` | `[nan, 0.0]` | none |
| `w102:448` | `fill.Scalar(n1,-2)` | `[-2]` | `+n4` (only one that lowers of 5) | `[0.93408]` | `n0`,`n1`,`n2` (3 candidates, 1 element) |
| `w103:514` | `fill.Scalar(n2,8)` | `[8,8,8,8]` | `+n5` | `[0.01475, 0.02054, 0.00022, -0.0]` | none |
| | | | `+n6` | `[0,0,0,0]` | none |
| `w100:389` | `min.unary_out` | `[0.47705]` | **ALONE** | **`[0.47705]` OK 3/3** | — |
| | | | `+n3`,`+n6`,`+n7` | `[0.47705]` **OK 3/3** | — |
| | | | `+n5` (`fill.Scalar`) | `[2e-05]` **MISMATCH 3/3** | none |

Readings:

- **`w100:66` holds up and is stronger than claimed** — the L0 match is genuinely byte-exact
  (verified with `torch.equal`, not 5% tolerance) and unique. But it fires with only **2 of 4**
  siblings; the other two return unmatched `[nan, 0.0]`. The write-up reports one value and hides
  this sibling-conditionality.
- **`w102:448` is vacuous.** A 1-element output matching 3 tensors carries ~no information — and the
  matched value **is `fill`'s own input** `n1`. "`fill` returned `self`" (a mis-lowering that drops
  the scalar operand) explains it at least as well as memory-planner aliasing. Same for
  `w100:66`'s `L0` under a chained-identity reading (`le.Scalar_out` -> `fill` both pass-through).
  **The constant-output probe cannot distinguish these two mechanisms**, and the bug file's "the
  **only** possible cause is that the memory planner aliased the fill's output buffer" is a false
  dichotomy.
- **`w103:514` supports nothing.** Two different garbage outputs, neither matching any buffer.
- **`w100:389` is a textbook confirmed graph-opt bug** — and the *alone* control that the write-up
  implies is unavailable is in fact available and **passes 3/3**. Claim 3 is vindicated; only the
  reported value (`0.0` vs `2e-05`) and the framing need fixing.

### 3.2 New 300-case gold-standard sweep **[V]**

Fresh uniform random sample of 300 of the 2,936 GRAPHOPT rows (seed 20260823), protocol
= **ALONE control x3 reps**, then each sibling **x3 reps**, requiring *stable* verdicts
(`tmp/.../audit_sweep.py`, 40 shards, ~7 min wall). Finite-population 95% CIs.

| class | n | share | 95% CI |
|---|--:|--:|---|
| **GRAPHOPT_CONFIRMED** (alone OK 3/3, >=1 sibling MISMATCH 3/3) | **105** | **35.0%** | 29.9-40.1 |
| &nbsp;&nbsp;. delta > 10x tolerance (gross, not precision) | 101 | 33.7% | 28.6-38.7 |
| &nbsp;&nbsp;. device output **all-zero (ZEROED / dropped store)** | 41 | 13.7% | 10.0-17.4 |
| &nbsp;&nbsp;. byte-exact match to some buffer | 24 | 8.0% | 5.1-10.9 |
| &nbsp;&nbsp;. **informative UNIQUE buffer identification** | **2** | **0.7%** | 0.0-1.5 |
| NOLOWER_ALONE, constant-output op probed **WRONG** (`fill` family) | 43 | 14.3% | 10.6-18.1 |
| &nbsp;&nbsp;. byte-exact to a unique buffer | 7 | 2.3% | 0.7-4.0 |
| NOLOWER_ALONE, not a constant-output op (**untestable**) | 76 | 25.3% | 20.7-30.0 |
| NOREPRO (alone OK, every sibling OK) | 33 | 11.0% | 7.6-14.4 |
| ALONE_ERR (emulator CRASH/SKIP on the alone run) | 21 | 7.0% | 4.3-9.7 |
| **CONFOUND_ALONE** (wrong with NO sibling — **not graph-opt**) | 12 | 4.0% | 1.9-6.1 |
| ERR (harness exception) | 8 | 2.7% | 0.9-4.4 |
| ALONE_UNSTABLE (alone flips run-to-run) | 2 | 0.7% | 0.0-1.5 |

**Good news for the artifact:** 105/300 = **35% of GRAPHOPT verdicts are gold-standard confirmed**
sibling-dependent divergences, with stability at 3/3 on both sides. Only **4 of 105 (3.8%)** sit
within 10x the (loose) tolerance, so these are **not** precision flips. The bisection's GRAPHOPT
label is largely trustworthy: only **4%** turn out wrong-alone.

**Bad news for the "one buffer-aliasing bug" framing:** applying an informativeness filter to the
byte-exact matches (>=2 elements, >=2 distinct values, not all-zero, not bool-shaped, unique match):

| byte-exact matches in the 105 confirmed | n | % |
|---|--:|--:|
| no match at all | 81 | 77.1 |
| matched, but device output **all-zero** (any zero tensor matches) | 11 | 10.5 |
| matched, but output is a **single scalar** | 9 | 8.6 |
| matched, but output constant / bool-shaped | 2 | 1.9 |
| **UNIQUE, informative identification** | **2** | **1.9** |

The only two informative ones are `w58:86` (`relu.default`) and `w27:221` (`cumsum.out`), both
returning `[-0.48249, 1.11035]` = the input leaf `L0`, byte-exact and uniquely. Together with
`w100:66` that is a **real, reproducible, cross-job signature** (3 independent jobs) — worth
reporting as such. But it is **~2-3% of the population, not 56%.**

Caveat: because every corpus graph uses `torch.manual_seed(12648430)`, a `(2,)` fp16 leaf has the
same values in *every* job, so "matches `L0`" is slightly weaker than it looks. **[I]**

**What the data actually supports as the headline mechanism:** **39% of the confirmed cases return
an all-zero output** — a **dropped store**, not a verbatim buffer copy. This matches this project's
own VGF `corpus_v4` finding ("graph-opt DROPPED-STORE (239 ZEROED) is the headline bug") and the
CoreML multi-op finding ("ZEROED dropped-store on aliased outputs"). Reframing QNN's finding as
DROPPED-STORE/ZEROED makes it **cross-validated on three independent backends** — a materially
stronger paper claim than the under-supported "verbatim buffer aliasing".

---

## 4. Confounds, quantified

### (a) fp16 device vs fp32 eager reference — **not the explanation for the confirmed cases [V]**

`WORKFLOW_RESULTS.md` is right that fp16-vs-fp32 drives the corpus-level mismatch rate (32.5% vs
Vulkan's ~11%), and **71.9%** of GRAPHOPT targets are float (38.3% fp16 + 33.6% fp32). But within
the 105 device-confirmed cases the deltas are gross, not marginal:

| max|delta| relative to the applied tolerance (rtol 1e-2 / atol 1e-3) | n | % |
|---|--:|--:|
| < 2x tol (borderline precision) | 1 | 1.0 |
| 2-10x tol | 3 | 2.9 |
| 10-100x tol | 30 | 28.6 |
| 100-1e4x tol | 49 | 46.7 |
| > 1e4x tol | 3 | 2.9 |
| non-finite (nan/inf structure differs) | 19 | 18.1 |

**fp16 precision can explain at most ~4% of the confirmed graph-opt population.** It plausibly
explains much of `NOREPRO` (11%) and `CONFOUND_ALONE` (4%). The write-up's "~250 small-delta
mismatches are fp16 noise" is not contradicted; it is simply not where the GRAPHOPT mass lives.

### (b) int64 `2^63` sentinel — **bounded and small [V]**

Target-output dtype over all 2,936 GRAPHOPT rows: fp16 1,125 (38.3%), fp32 986 (33.6%), uint8 311
(10.6%), **int64 153 (5.2%)**, bool 138 (4.7%), int16 106 (3.6%), int32 100 (3.4%), int8 17 (0.6%).
So the sentinel confound is capped at **5.2%**. In the confirmed sample, exactly **1 of 105 (1.0%)**
shows a >=1e18 magnitude (`w127:405` `pow.Tensor_Scalar_out`, device `9.339e19`) and should be
excluded, as the Ethos-U review excluded `w36:1`.

A *new* confound this audit surfaced: **53 of 105 (50.5%)** confirmed cases have device outputs whose
values lie entirely in `{0, 1, 253, 254, 255}`, and two (`w118:210`, `w48:182` `gt.Scalar_out`)
return literal `253`/`255` where the reference is `0`/`1`. `uint8`+`bool` are 15.3% of GRAPHOPT
targets. This smells like a **bool-as-byte (`0xFF` vs `1`) representation mismatch**, which is a
dtype-marshalling defect, not memory planning. **Not yet investigated — flagging it. [I]**

### (c) Portable fallback / is it really QNN? — **partially [V]**

All 105 confirmed cases have `delegated_ops >= 1` — **no case is pure portable fallback**, so the
"QNN-delegated" premise holds. But comparing `delegated_ops` against the subgraph's cone op count:

- **fully delegated (`delegated_ops >= n_cone_ops`): 42 / 105 (40%)**
- **partial — portable kernels co-resident: 63 / 105 (60%)**
- `delegated_ops == 0`: 0

So for **60%** of the confirmed cases the graph is a *mix* of QNN delegate blobs and ExecuTorch
portable kernels, and the delegate/portable buffer boundary is an equally plausible defect site.
Attributing the bug specifically to "**QNN's** memory planner" is therefore **not established** for
the majority — it may be an ExecuTorch partition-boundary/memory-planning defect that QNN merely
exposes. Suggestive detail: in `w100:66` the byte-exact `L0` clobber occurs precisely in the two
*partially* delegated configurations (`deleg_ops=2` for `+n4`/`+n6`), while the two *fully*
delegated ones (`deleg_ops=3`/`4`) return unmatched `[nan, 0.0]`. **[I]**

Caveat: this project's own memory (`delegation-metric-plumbing-inflation`) records that
`delegated_ops` over-counts, so treat the 40/60 split as a proxy. **[I]**

### (d) `executor/compare.py::select` silent fallback — **checked, NOT implicated [V] (negative result)**

`select` does silently return *all* outputs when `user_pos` is out of range, and this hazard did
break the VGF SHAPE category. It does **not** apply here:

- `user_pos` is derived from the exported program's own specs
  (`gen/export/job.py:151`: `[i for i,s in enumerate(specs) if s.kind == OutputKind.USER_OUTPUT]`),
  so it cannot index outside the program's output list.
- Verified on device: every 2-output subgraph tested reports `user_pos=[0,1]` with `n_eager=2`
  (no prepended mutation outputs), and the full `w100:66` job header carries
  `user_pos=[0,1,2,3,4]` for 5 eager outputs — the identity list.
- `sweep_batch.py`'s subsequent `et[len(et)-len(eager):]` trim is therefore a no-op.

The "fill returned the input leaf `L0`" observation is **not** an output-slot misalignment artefact.

---

## 5. Precedent comparison — which defects recur

| defect from the precedents | present in the QNN classifier? |
|---|---|
| **Stage-count conflation** (Ethos-U: "21 DRIFT + 5 CLOBBER is the *pinpoint* stage") | **YES** — "`fill.Scalar` ... 520 cases" is the all-mismatch count, not the GRAPHOPT count (§1.4). |
| **Test passes trivially on degenerate data** (Ethos-U: constant reference) | **YES, badly** — the clobber match is vacuous on 1-element and all-zero outputs; 22 of 24 byte-exact matches in the confirmed set fail an informativeness filter (§3.2). |
| **Nondeterministic rows carried into the category** (VGF: `n_div=1/3`) | **YES, systemically** — one run per pair on the project's least-deterministic target; a no-sibling-effect model reproduces the split (§2.2). |
| **Bare / low-power test** (VGF: element-count inequality) | **YES** — no ALONE control; `nsk` conflates build-SKIP, device-SKIP, CRASH and TIMEOUT; demonstrably wrong on `w100:389` (§2.1). |
| **`select()` silent fallback** (VGF) | **NO** — structurally excluded (§4d). |
| **Raw device values never archived** (Ethos-U) | **PARTLY** — the sweep TSVs store only `nok/nmm/nsk`, no deltas or device values, so nothing could be re-adjudicated without a re-run. |
| *new* — **unfalsifiable probe** (match -> aliasing; no match -> aliasing) | **YES** (§2.4). |

Where QNN is **better** than both precedents: the sample is a genuine 1-in-5 stride (§1.3); the
arithmetic is clean (§1.1-1.2); and — unlike VGF SHAPE (0/12 real) and Ethos-U REQUANT (1/3 real) —
**the underlying phenomenon survives re-verification at scale (105/300)**. This category does *not*
collapse. Only the mechanism attribution and the 56% figure do.

---

## 6. What a defensible set of numbers looks like

| statement | n/300 | share | 95% CI |
|---|--:|--:|---|
| GRAPHOPT verdicts that are **device-confirmed sibling-dependent** (alone OK 3/3, sibling MISMATCH 3/3) | 105 | **35.0%** | 29.9-40.1 |
| . with **gross** (non-precision) deltas | 101 | 33.7% | 28.6-38.7 |
| . whose signature is **ZEROED / dropped store** | 41 | 13.7% | 10.0-17.4 |
| Upper bound incl. alone-untestable constant-output ops probed wrong | 148 | 49.3% | 44.0-54.7 |
| **Positively identified verbatim buffer-clobber** ("the one aliasing bug") | ~7-9 | **2.3-3.0%** | 0.7-4.7 |
| Mislabelled by the bisection (wrong with no sibling) | 12 | 4.0% | 1.9-6.1 |
| Flaky / non-reproducing | 33 | 11.0% | 7.6-14.4 |
| Not classifiable (untestable alone, or emulator crash/error) | 107 | 35.7% | 30.5-40.8 |

Suggested re-wording:

> Of the 2,936 GRAPHOPT verdicts, a 300-case random sample re-tested with an alone control and 3
> repetitions confirms **35.0% [29.9-40.1]** as stable, sibling-dependent divergences with deltas
> 10-10^4x past tolerance — real graph-level defects, not fp16 noise (only 3.8% are within 10x
> tolerance). The dominant signature is a **dropped store**: 39% of confirmed cases return an
> all-zero output, matching the ZEROED graph-opt defect independently found on Arm VGF and CoreML.
> A verbatim buffer-clobber (device output byte-identical to a uniquely identified live buffer) is
> positively confirmed in ~2-3% of cases, including `w100:66`, `w58:86` and `w27:221`, which all
> return the input leaf `L0` byte-exactly. A further 14.3% are constant-output ops (`fill.Scalar`
> family) that cannot be lowered alone and are provably wrong on device, but whose mechanism
> (aliasing vs. mis-lowering to an identity) is undetermined. 4.0% were mislabelled (wrong without
> any sibling) and 11.0% do not reproduce.

Do **not** claim: one bug; ">=56%"; "byte-identical" for the 5%-tolerance matches; "the only possible
cause is aliasing"; or "`fill.Scalar` ... 520 cases".

---

## 7. Negative results / things that did *not* turn out to be problems

- **`compare.select` misalignment is not implicated** — `user_pos` comes from `OutputKind.USER_OUTPUT`
  and is the identity list in every case inspected (§4d). The "fill returns `L0`" finding is real,
  not a slot-indexing artefact.
- **The 587 is a proper systematic random sample**, not a convenience prefix (§1.3). The
  extrapolation to 2,936 is sound; only what was *measured* is at fault.
- **The arithmetic reconciles to 100.01%** with no double-counting (§1.2).
- **The bisection's GRAPHOPT label is mostly sound** — only 4.0% are wrong-alone (§3.2). The
  WORKFLOW_RESULTS hedge that GRAPHOPT verdicts "still over-count" is more pessimistic than the data.
- **int64 `2^63` sentinels are a small confound here** (5.2% ceiling, 1.0% observed) — unlike the
  Ethos-U case where they invalidated a headline row (§4b).
- **Every confirmed case is genuinely QNN-delegated** (`delegated_ops >= 1`, zero exceptions) — the
  portable-fallback worry does not void the finding, though it does complicate attribution (§4c).
- **The category does not collapse.** Unlike VGF SHAPE (0/12) and Ethos-U REQUANT (1/3), 105 of 300
  survive the strictest test available. This is a real finding that is currently over-claimed, not a
  phantom.
- **The emulator is cheap to re-run** (~1 s/inference, no broker needed) — there is no reason to ship
  the weaker sibling-only evidence. `tmp/pinpoint_graphopt.py` already implements the right protocol
  and only needs `BK`/`CO` switched from `ethos-u` to `qualcomm`.

### Unverifiable now **[U]**
- The provenance of the "min correct with 3 of 4 siblings" claim: no `sibling_sweep.py` log for
  `w100:389` was preserved anywhere in `tmp/`. I re-derived it on device (and it holds), but the
  original evidence is gone.
- Whether the sweep's `nsk` entries were timeouts vs. build-skips: not recorded per row.
- The `uint8`/bool `253`/`255` representation class (§4b) is newly flagged and un-investigated.

### Reproducing this audit
All scripts and result TSVs are preserved in **`tmp/qnn_audit/`**:

| file | what it does |
|---|---|
| `audit_sweep.py` | the gold-standard protocol (ALONE control x3 + each sibling x3) — produced `audit_all.tsv` |
| `delta.py` | delta magnitude, delegation split, byte-exactness — produced `delta_all.tsv` |
| `reverify.py` | the four named jobs (`w100:66`, `w102:448`, `w103:514`, `w100:389`), 3 reps each |
| `binom.py`, `bb.py` | the binomial and beta-binomial no-sibling-effect nulls (§2.2) |
| `hostvals.py` | host-side tensor values + clobber-match uniqueness for the named jobs |
| `int64.py` | GRAPHOPT target dtype histogram (§4b) |
| `sweep.tsv` | the original 587-case sweep, de-mangled from `tmp/sweep_shards/*.tsv` (§1.2) |
| `audit_cases.tsv` | the 300-case random sample (seed 20260823) |

Run them with `source tmp/qnn_env.sh; export MOBILE_BACKENDS=qualcomm CUDA_VISIBLE_DEVICES=
PYTHONPATH=/data/jwen929`, sharded via `SHARD_ID`/`NUM_SHARDS` (40 shards ~= 7 min for 300 cases).
No broker is required — `qnn_client.QnnExecutor` is driven in-process.
