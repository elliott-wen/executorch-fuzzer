# `VALUE` / `SCALE:*` / `ALIAS:*` audit — device-verified re-classification

**Target:** the three least-verified mechanism buckets of `findings/vgf/graphopt_all.tsv`:
`VALUE` (332 rows), `SCALE:*` (24 rows), `ALIAS:*` (27 rows).
`GRAPHOPT_REPORT.md` §4 reads the 275 `COMPOSITIONAL`+`AB_OK` `VALUE` rows as
**"fp16 number-format divergence"**. That reading is **not supported by the data.**

`VALUE` is a *residual*: `tmp/vgf_go_deep.py::mech()` returns SHAPE / NONFINITE / ZEROED /
SCALE / ALIAS by five specific tests and falls through to `VALUE` when none fire. `VALUE`
therefore means "none of the above", and the bucket is a mixture.

Everything below was produced by **running** the reconstructed A/B graphs on the VGF runtime
(and on `portable`/`xnnpack`). Nothing is inferred from the prose of the earlier report.
Where a claim is inferred rather than measured it is marked *(inferred)*.

---

## 1. Method and commands

Sample: **93 cases** — 42 `VALUE` rows drawn at random with `random.seed(20260823)`
(30 of the 277 `COMPOSITIONAL`, 12 of the 55 `SIBLING_DEPENDENT`; stratified, hence
*disproportionate* — bucket-level percentages in §4 are re-weighted to 277:55), plus **all
24 `SCALE:*` and all 27 `ALIAS:*` rows**. `needed_siblings` / `needed_live` and `deleg_full`
were joined back in from `findings/vgf/bisect_results.tsv`.

Environment (without `MODEL_CONVERTER_LIB_DIR` the model-converter dies on `GLIBCXX_3.4.30`,
`get_backend("vgf")` returns `None`, and every `build_job(src,"vgf",False)` returns
`SKIP: unknown backend 'vgf'` — i.e. a silent total no-op):

```bash
export PYTHONPATH=/data/jwen929
export CUDA_VISIBLE_DEVICES=
export MODEL_CONVERTER_LIB_DIR=/opt/nvidia/nsight-systems/2024.6.2/host-linux-x64
export PATH=/data/jwen929/mobile/.venv/bin:$PATH
# precondition actually checked first:
python -c 'from mobile.gen.export import get_backend; print(get_backend("vgf"))'
#   -> <mobile.gen.export.backends.vgf.VgfBackend object at 0x...>       (non-None: OK)
```

Per case, exactly the A/B contrast `vgf_go_deep.py` claims to make:

* `COMPOSITIONAL`   — A = `cone_localizer(py,out)[1]([])`, B = `...(needed_live)`
* `SIBLING_DEPENDENT` — A = `output_set_localizer(py,out)[1]([target])`,
  B = `...([target]+needed_siblings)`, target read at `sel.index(target)`

each lowered with `build_job(src,"vgf",False)` and run on the real runtime with
`mobile/vgf_client/vgf_runner.sh --pte … --input … --out DIR`, outputs read back from
`DIR/out_<i>.bin` and diffed with `mobile.executor.compare._cmp`. **A × 2 reps, B × 3 reps.**
The whole 93-case sweep was then **run twice, end to end, in independent sessions**
(6 shards each) — see §2.

```bash
# sample construction, device sweep (6 shards), offline re-classification, cross-backend
python sample.py  cases.json                        # 93 cases
python audit.py --cases cases.json --out audit_$i.tsv --shard $i --nshards 6   # i=0..5
python reclass.py                                   # corrected classifier over the dumps
bash   xb_all.sh  $W $i                             # portable + xnnpack, one per subprocess
```
Probe sources are kept in `findings/vgf/value_probe/` (copies of the scratch scripts).

### The corrected classifier
Tests in this order, and **which test fired is recorded** (`test` column):

| test | label | rule |
|---|---|---|
| t0 | `MISALIGN_OR_SHAPE` | `ref.shape != dev.shape` |
| t1 | `NONFINITE` | `torch.isfinite(ref) != torch.isfinite(dev)` anywhere, **or** matched non-finites that differ (`-inf` vs `+inf`, `nan` vs `inf`). The original test only ever compared `isnan`, so **every infinity bug fell through to `VALUE`.** |
| t2 | `EXTREME_MAGNITUDE` | `max(|ref|,|dev|) >= 2^48`; the exact power of two is reported (the `2^63` int64 sentinel lands here) |
| t3 | `FP16_SATURATION` | some `|dev| == 65504` where `|ref| != 65504` |
| t4 | `ZEROED` | dev all ~0, ref not |
| t5 | `PARTIAL_ZERO` | some lanes zeroed, **all remaining lanes match** |
| t6 | `CLOBBER_EXACT/NEAR:<node>` | dev byte-identical (or `allclose` 1e-3/1e-4) to another **live tensor of the graph** |
| t4b | `REF_ZERO_DEV_NONZERO` | ref all ~0, dev not (the inverse of `ZEROED`; the original classifier had no test for it) |
| t7 | `PERMUTED` / `SHIFTED` | **same multiset, wrong positions** (sorted-`allclose` but unsorted not) — no such test existed |
| t7b | `MISPLACED_ELEMS` | ≤ max(2, n/4) lanes wrong, and every wrong lane holds a value found elsewhere in `ref` |
| t8 | `SCALE:<r>` | consistent ratio **AND ≥ 8 surviving ratios AND `ref.std() > 0.01*ref.abs().mean()`**. Without both guards the test is vacuous — the failure mode already documented in `findings_v2/ethos_u_fvp/REQUANT_SCALE_REVIEW.md` |
| t9 | `NUMERIC` / `INT_WRONG_VALUE` | residual; reports `max|Δ|/|ref|` per element **and** `max|Δ|/max|ref|` (the point-wise ratio explodes to 1e12 on a near-zero lane and is useless on its own) |

---

## 2. Does it reproduce? — yes, completely, and deterministically

| | count |
|---|---|
| cases attempted | 93 |
| A correct **2/2** and B divergent **3/3** (`AB_OK`) | **93 / 93 (100%)** |
| flaky (A correct, B intermittent) | **0** |
| `A_ALSO_DIVERGES` (operator bug, not graph-opt) | **0** |
| build failures (`build_job` != READY) | **0** |
| run failures | 0 (1 harness bug of my own — see below) |

The whole sweep was run **twice** in independent sessions. Verdict *and* mechanism label agreed
on **93/93** cases. Counting both sessions that is **A correct 4/4 and B divergent 6/6 for every
case.** This bucket is not the SHAPE bucket: the A/B contrast is real and stable.

**Cross-tab against recorded `n_div`:** vacuous here — **every** row in all three buckets was
recorded at `n_div = 3` (`awk -F'\t' '$9=="VALUE"{print $6,$7}' graphopt_all.tsv` → `330 AB_OK 3`,
`2 A_ALSO_DIVERGES 3`; same for `SCALE:*`/`ALIAS:*`). The `n_div=1` flakiness that accounted for
4/12 of the SHAPE bucket simply does not occur here. The two recorded `A_ALSO_DIVERGES` `VALUE`
rows were excluded from the sampling pool by construction, as they should be.

*Negative result, recorded:* the single `RUN_FAIL` in the first pass (`w127:717`) was **my** bug,
not the artifact's — the B graph has a legitimate `(0,0,0)` int64 second output, and
`torch.frombuffer` refuses a 0-byte buffer. Fixed; the case then classified as `NONFINITE`.

## 3. What it actually is

Recorded label → measured mechanism, 93 cases:

| recorded | n | measured decomposition |
|---|---|---|
| `VALUE` | 42 | `NUMERIC` 12 · `EXTREME_MAGNITUDE` 7 · `INT_WRONG_VALUE` 7 · `REF_ZERO_DEV_NONZERO` 7 · `NONFINITE` 5 · `PARTIAL_ZERO` 4 |
| `SCALE:*` | 24 | `NUMERIC` 9 · **`SCALE` 4** · `CLOBBER_NEAR` 4 · `INT_WRONG_VALUE` 3 · `CLOBBER_EXACT` 2 · `EXTREME_MAGNITUDE` 1 · `SHIFTED` 1 |
| `ALIAS:*` | 27 | **`CLOBBER_NEAR` 15 · `CLOBBER_EXACT` 10** · `NONFINITE` 1 · `PARTIAL_ZERO` 1 |

`VALUE` split by verdict (COMPOSITIONAL n=30 / SIBLING_DEPENDENT n=12):

* COMPOSITIONAL: `NUMERIC` 10 · `EXTREME_MAGNITUDE` 7 · `INT_WRONG_VALUE` 6 · `REF_ZERO_DEV_NONZERO` 6 · `PARTIAL_ZERO` 1
* SIBLING_DEPENDENT: `NONFINITE` 5 · `PARTIAL_ZERO` 3 · `NUMERIC` 2 · `REF_ZERO_DEV_NONZERO` 1 · `INT_WRONG_VALUE` 1

**Reading:** at most 12/42 (28.6%, 95% CI [17.2, 43.6]) of `VALUE` is even *arguably* numeric
before cross-backend filtering; 30/42 is non-finite structure, integer wrongness, zero/non-zero
structure, or 2^n-magnitude garbage. `ALIAS:*` is the one recorded label that **survives as
stated** (25/27 re-confirm as a clobber) — but see the vacuity caveat in §6.
`SCALE:*` survives in only **4/24** cases.

## 4. Bucket-level decomposition (re-weighted to the real 277:55 split)

Point estimates for the whole 332-row `VALUE` bucket, weighting the two strata by their
population share. n=42, so these carry ±10–15 pp of sampling error — treat them as **shape of
the mixture, not precise counts.**

| measured mechanism | share of `VALUE` | ≈ rows of 332 |
|---|---|---|
| `NUMERIC` (float, residual) | 30.6% | ~101 |
| `EXTREME_MAGNITUDE` (2^n garbage) | 19.5% | ~65 |
| `INT_WRONG_VALUE` (integer, exact-tol) | 18.1% | ~60 |
| `REF_ZERO_DEV_NONZERO` | 18.1% | ~60 |
| `NONFINITE` (missed by the old isnan-only test) | 6.9% | ~23 |
| `PARTIAL_ZERO` | 6.9% | ~23 |

And after the cross-backend and delegation filters of §5–6:

| | share of `VALUE` | ≈ rows |
|---|---|---|
| reproduces identically on `portable` and/or `xnnpack` (**not VGF, not graph-opt**) | 54.2% | ~180 |
| VGF-specific (portable/xnnpack agree with eager) | 31.9% | ~106 |
| inconclusive (both host backends error out) | 13.9% | ~46 |
| B graph contains **no VGF delegate at all** (`delegated_ops == 0`) | 30.6% | ~102 |
| **VGF-specific AND actually delegated** | **31.9%** | **~106** |
| …of which residual `NUMERIC` | 15.3% | ~51 |

## 5. Is "fp16 number format" tenable? — **No. Zero of 21 cases.**

fp16 has ~10 bits of mantissa: relative error from number format is ~5e-4, at most ~1e-3.
The 21 cases the corrected classifier calls `NUMERIC`, sorted by `max|Δ| / max|ref|`:

```
0.0129 0.0155 0.0362 0.0574 0.112 0.217 0.273 0.289 0.316 0.325 0.344
0.398  0.438  0.566  0.750  0.753 1.00  1.05  2.00  2.08  8.00
```

| relative error | count |
|---|---|
| ≤ 1e-3 (fp16 rounding) | **0 / 21** (95% CI [0, 15.5]%) |
| 1e-3 – 1e-2 | **0 / 21** |
| > 1e-2 | **21 / 21** |

The *smallest* observed error is **1.3e-2 — thirteen times** the fp16 ceiling, and it is a `tan`
near a pole (ill-conditioned, so even that one is more plausibly an amplified upstream input
difference than a rounding effect). The median is ~0.3, i.e. **30% wrong.** Seven cases are
≥ 100% wrong. `min|Δ|` is not the issue either: the small-magnitude cases have *exact-integer*
deltas (`4.617 → 3.617`, `-8 → -2`, `258 → 2`), which rounding cannot produce.

**"fp16 number-format divergence" is refuted for the whole sampled `NUMERIC` sub-bucket.**
Looking at the actual values, the residual "numeric" cases are mostly *wrong-input* or
*wrong-branch* bugs wearing numeric clothing:

| case | target | trigger | rel | values |
|---|---|---|---|---|
| w61:506:3 | `tan` | `t_copy;native_group_norm;abs` | 0.0129 | ref=[6.0586, 5.0156, 1.0869] dev=[6.0586, 5.0938, 1.0879] |
| w48:634:0 | `t_copy` | `asinh` | 0.112 | ref=[-3.377, 1.1104] dev=[-3.7539, 1.1104] |
| w26:67:0 | `sub` | `floor_divide` | 0.217 | ref=[4.6172, 4.6172, 4.6172, 4.6172] dev=[**3.6172**, 4.6172, 4.6172, 4.6172] — one lane off by **exactly 1.0** |
| w63:310:2 | `_softmax` | `max` | 0.398 | ref=[0.1689, 0.8311] dev=[**0.5, 0.5**] — softmax collapsed to uniform, i.e. its two input lanes became equal |
| w98:638:1 | `div` | `sum` | 0.438 | ref=[0.0869, 0.0869] dev=[0.125, 0.125] |
| w99:467:1 | `asin` | `sign;floor_divide;_softmax` | 0.566 | ref=[0.5234, 0.5234] dev=[0.2727, **0.8198**] — two equal lanes became unequal |
| w37:595:0 | `rsub` | `unbind_copy` | 0.75 | ref=[-8.0, -8.0] dev=[-2.0, -2.0] — exact integers |
| w77:456:1 | `abs` | `atan2;remainder;lift_fresh_copy` | 1.00 | ref=[0.7852…]=π/4 dev=[1.5703…]=**π/2** |
| w69:691:2 | `atan2` | `elu` | 2.00 | ref=[**-3.1406**, 0.7852] dev=[**+3.1406**, 0.7852] — **sign of π: atan2 branch cut** |
| w26:379:3 | `cos` | `index;squeeze_copy;elu` | 2.08 | ref=[-0.0597, -0.0597] dev=[-0.1837, -0.1837] |

A defensible name for this group is **wrong operand / wrong branch**, not "number format".
Two of the ten (`w69:691` π sign flip, `w77:456` π/4→π/2) are recognisably `atan2` quadrant
bugs; three (`w26:67`, `w63:310`, `w99:467`) are lane-level data errors that happen to be
observed through a smooth op. *(The upstream attribution is inferred from the values; I did not
bisect each one to the producing node.)*

## 6. Confounds found

**(a) Portable fallback — present and large.** `deleg_full >= 1` was enforced on the *original*
corpus job, but the A/B graphs are *reconstructions*, and the reconstruction frequently destroys
the delegate. Measured on the rebuilt graphs:

* **23 / 93 (24.7%)** of B graphs have `delegated_ops == 0` — nothing runs on VGF at all.
* **36 / 93** of A graphs have `delegated_ops == 0`, 13 of them paired with a delegating B — so
  in those the "A is correct / B is wrong" contrast is *portable vs VGF*, not a VGF plan change.
* **23 / 23** of the `delegated_ops == 0` B cases reproduce on `portable`/`xnnpack`. 100%.
  This is a clean internal consistency check on the whole cross-backend method.

**(b) The int64 `2^63` sentinel — present, and it is a single non-VGF operator bug.** All 8
`EXTREME_MAGNITUDE` cases have `bitwise_left_shift` among the trigger ops; the device values are
exactly `2^63`, `2^61`, `2^60`, `2^56`, … against an eager reference of `0`. Eager torch returns
0 for an out-of-range shift; the ExecuTorch kernel performs the C++ UB shift. Every one is
**byte-identical on `portable`, `xnnpack` and VGF.** Widening the net: **19 of 93 cases have a
`bitwise_*_shift` trigger op, and 19/19 (100%) are SHARED with the host backends.** One recorded
`SCALE:-7720456580759552.000` is precisely this: `ratio = -7.72e15` against `ref = 56` is
`-4.32e17/56`, i.e. UB-shift garbage, not a scale factor.

**(c) Output misalignment (`compare.py::select` silently returning ALL outputs when `user_pos`
is out of range) — checked, and NOT present.** *Negative result.* The probe flags every
`user_pos` fallback; across all 93 × 5 device runs the flag fired **0 times**
(`awk -F'\t' '$21!=""' all2.tsv | wc -l` → 0). `n_pte_out == len(eager)` and `user_pos` is a
valid in-range index list in every case. So unlike the SHAPE bucket, misalignment does not
explain anything here. (The `select` fallback is still a latent hazard and should raise.)

**(d) New confound: vacuous `ALIAS`/`CLOBBER` on tiny tensors.** The `ALIAS` label re-confirms
25/27 times — but the test "dev equals some other live tensor" is nearly free on a 1–2 element
tensor. Of 31 clobber verdicts, **22 are vacuous** (`numel < 8`, or more than one live tensor
matches): 18 have `numel <= 2`, and in 8 of those the device value is simply `1.0`, which some
node in almost any graph holds. Requiring `numel >= 8` **and** a unique match leaves **9 of 31**
defensible clobbers (`w20:329`, `w23:9`, `w25:146`, `w4:391`, `w53:531`, `w57:285`, `w58:331`,
`w66:273`, `w93:212`) — and 5 of those 9 are SHARED with the host backends, leaving **3**.
This is the same vacuity failure the `REQUANT_SCALE_REVIEW` found in `SCALE`, in a second test.

**(e) Point-wise relative error is itself a confound.** `max(|Δ|/|ref|)` reaches 4.8e11, 1.6e13,
9.2e30 in this data purely because some `ref` lane is 0 or denormal. Any relative-error figure
quoted from this artifact must be `max|Δ| / max|ref|` or it is meaningless.

## 7. Is any of it VGF-specific? — half is not

Each reconstructed **B** graph was re-lowered to `portable` and to `xnnpack` and run in-process
via `mobile.executor.et_runner.run_pte`, one backend per subprocess.

| | count of 93 |
|---|---|
| **SHARED** — `portable` and/or `xnnpack` also MISMATCH → **not a VGF finding at all** | **47 (50.5%)** |
| **VGF_ONLY** — host backends match eager, VGF does not | 38 (40.9%) |
| INCONCL — both host backends raise `error 0x12` (`InvalidArgument`) on the rebuilt graph | 8 (8.6%) |

Of the 47 SHARED cases, **44 have a device output byte-identical (or `allclose` 1e-3/1e-4) to `portable`'s** — same wrong bits,
not merely a same-direction error. By recorded bucket (SHARED / VGF_ONLY / INCONCL): `VALUE` **20 / 16 / 6**,
`SCALE:*` **12 / 11 / 1**, `ALIAS:*` **15 / 11 / 1**. Per-case figures in
`findings/vgf/value_audit_cases.tsv`.

GRAPHOPT_REPORT already concedes that non-finite saturation is shared with xnnpack/openvino. The
measurement here says the sharing is far broader: **half of these three buckets is a plain
eager-vs-ExecuTorch divergence that has nothing to do with VGF or with any graph optimisation.**

## 8. Verdict: what survives

Funnel over the 93 sampled cases:

```
93  sampled                          (VALUE 42, SCALE:* 24, ALIAS:* 27)
93  reproduce  A 2/2 correct, B 3/3 divergent, stable across two independent sessions
-23  B graph has NO VGF delegate  (delegated_ops == 0; 23/23 SHARED)
-47  SHARED with portable and/or xnnpack   (44 byte-identical)   [overlaps the 23]
 -8  inconclusive (host backends error 0x12)
=38  VGF-specific AND delegated
-10  label vacuous on inspection (tiny-tensor CLOBBER, mostly)
=28  fully defensible graph-opt-conditioned VGF divergences
```

The 28 survivors: `NUMERIC` 10 · `NONFINITE` 4 · `PARTIAL_ZERO` 4 · `SCALE` 3 ·
`REF_ZERO_DEV_NONZERO` 2 · `CLOBBER_EXACT` 2 · `INT_WRONG_VALUE` 2 · `CLOBBER_NEAR` 1.

**Explicit verdict on a "numeric plan" / "fp16 number format" category: it does not survive.**

* Scaled to the bucket, the `VALUE` rows that are VGF-specific **and** delegated are ≈32%
  (16/42 sampled, 95% CI [25, 53]) → **~106 of 332**, not 275.
* Of those, the ones that are residual-`NUMERIC` at all are 6/42 sampled → **~51 of 332**.
* Of the 21 `NUMERIC` cases measured, **0 have relative error at fp16-rounding magnitude.**
  The minimum is 1.3e-2. There is **no** case whose divergence is explained by number format.
* The 10 VGF-specific delegated `NUMERIC` survivors are better named **wrong operand / wrong
  branch** (§5): sign-of-π `atan2` flips, exact-integer lane offsets, and softmax/asin lane
  collapse. That is a real and reportable finding — it is just not a number-format finding, and
  it is an order of magnitude smaller than the 275 claimed.

**Recommended reportable numbers.** For `VALUE`: *"332 rows; a 42-row device-verified sample
reproduces 100% but decomposes into six mechanisms, of which ~54% also reproduce on portable/
xnnpack; ≈16/42 (~106 rows) are VGF-specific and delegated, and 0/21 numeric cases are within
fp16 rounding."* For `SCALE:*`: **4 of 24 survive the guarded test, 3 of those are VGF-specific**
(`w114:621` int32 ×4.22e6, `w29:743` fp16 ×0.5, `w74:560` fp32 ×0.75; `w70:715` ×124 is SHARED).
For `ALIAS:*`: **25/27 confirm as a clobber, but only 9 non-vacuously and only 3 of those are
VGF-specific.** Delete the "fp16 number-format divergence" claim from §4 or replace it with the
§5 table.

## 9. Suggested fixes to the classifier
1. `mech()` t1 must use `torch.isfinite`, not `isnan` — infinities currently fall into `VALUE`
   (measured: 6 of 93 sampled cases, ~7% of `VALUE`).
2. Add the missing tests: `REF_ZERO_DEV_NONZERO`, `PARTIAL_ZERO`, `PERMUTED`/`SHIFTED`,
   `EXTREME_MAGNITUDE` (2^n), and a separate integer bucket (`INT_WRONG_VALUE`) — integer
   outputs are compared at tol 0 and can never be a "number format" story.
3. Guard `SCALE` with `n_ratio >= 8` **and** `ref.std() > 0.01*ref.abs().mean()`; guard
   `ALIAS`/`CLOBBER` with `numel >= 8` **and** a unique match.
4. Record `delegated_ops` **of the reconstructed A and B graphs**, and drop rows where
   `B.delegated_ops == 0` — 24.7% of these rows never touch the backend under test.
5. Run every candidate against `portable`/`xnnpack` before calling it a backend finding; half
   of this sample fails that check.
6. `executor/compare.py::select` should raise on an out-of-range `user_pos` instead of silently
   returning all outputs (not implicated here, but it was in the SHAPE bucket).

---

## 10. Per-case table

`repro` is "A correct 2/2, B divergent 3/3" for all 93 (and again in the second session).
`rel_err` is `max|Δ|/|ref|` per element / `max|Δ|/max|ref|`. Full machine-readable
version: `findings/vgf/value_audit_cases.tsv`.

| job | out | verdict | recorded | actual mechanism | test | repro (A ok / B div) | rel_err (max / scaled) | dtype, numel | B_deleg | portable | xnnpack | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w93:212 | 0 | COMP | `ALIAS:L0` | **CLOBBER_EXACT:L0** | t6 | 2/2, 3/3 | - | int64, 32 | 0 | MISMATCH | MISMATCH | tgt=`alias_copy` trig=`remainder`; ref=[-3.0, -3.0, -3.0] dev=[-3.0, -3.0, -3.0]; byte-identical to another live tensor |
| w65:110 | 3 | COMP | `ALIAS:L1` | **CLOBBER_EXACT:L1** | t6 | 2/2, 3/3 | - | int16, 1 | 3 | MISMATCH | MISMATCH | tgt=`bitwise_or` trig=`sum`; ref=[-3.0] dev=[-4.0]; VACUOUS(n=1,matches=1); byte-identical to another live tensor |
| w20:329 | 1 | COMP | `ALIAS:L2` | **CLOBBER_EXACT:L2** | t6 | 2/2, 3/3 | - | float16, 128 | 4 | OK | OK | tgt=`scatter_add` trig=`slice_copy;split_with_sizes_copy;_to_cop`; ref=[0.8169, 1.7637, 0.3711] dev=[0.0056, 1.7637, 0.3711]; byte-identical to another live tensor |
| w117:560 | 0 | COMP | `ALIAS:L3` | **CLOBBER_EXACT:L3** | t6 | 2/2, 3/3 | - | bool, 1 | 0 | MISMATCH | MISMATCH | tgt=`any` trig=`bitwise_left_shift`; ref=[0.0] dev=[1.0]; VACUOUS(n=1,matches=1); byte-identical to another live tensor |
| w117:596 | 0 | SIBL | `ALIAS:n1` | **CLOBBER_EXACT:n1** | t6 | 2/2, 3/3 | - | float16, 4 | 1 | OK | OK | tgt=`slice_copy` trig=`pow`; ref=[-0.0965, 0.222, 0.1868] dev=[-1.0, -1.0, -1.0]; VACUOUS(n=4,matches=3); byte-identical to another live tensor |
| w13:110 | 0 | SIBL | `ALIAS:n2` | **CLOBBER_EXACT:n2** | t6 | 2/2, 3/3 | - | float16, 1 | 5 | OK | OK | tgt=`mean` trig=`constant_pad_nd`; ref=[-0.5825] dev=[3.0]; VACUOUS(n=1,matches=1); byte-identical to another live tensor |
| w56:227 | 2 | COMP | `ALIAS:n3` | **CLOBBER_EXACT:n3** | t6 | 2/2, 3/3 | - | int64, 1 | 0 | MISMATCH | MISMATCH | tgt=`sign` trig=`remainder`; ref=[-1.0] dev=[1.0]; VACUOUS(n=1,matches=2); byte-identical to another live tensor |
| w82:146 | 2 | SIBL | `ALIAS:n4` | **CLOBBER_EXACT:n4** | t6 | 2/2, 3/3 | - | bool, 1 | 8 | OK | OK | tgt=`ne` trig=`div;clone`; ref=[0.0] dev=[1.0]; VACUOUS(n=1,matches=5); byte-identical to another live tensor |
| w29:122 | 2 | COMP | `ALIAS:n6` | **CLOBBER_EXACT:n6** | t6 | 2/2, 3/3 | - | uint8, 1 | 1 | OK | OK | tgt=`copy` trig=`_to_copy;slice_copy;_to_copy;slice_copy;`; ref=[252.0] dev=[253.0]; VACUOUS(n=1,matches=1); byte-identical to another live tensor |
| w66:273 | 0 | SIBL | `ALIAS:n7` | **CLOBBER_EXACT:n7** | t6 | 2/2, 3/3 | - | float32, 324 | 7 | OK | OK | tgt=`mul` trig=`ceil`; ref=[-0.0, -0.0, -0.0] dev=[1.0, 1.0, 1.0]; byte-identical to another live tensor |
| w23:9 | 0 | COMP | `ALIAS:L0` | **CLOBBER_NEAR:L0** | t6 | 2/2, 3/3 | - | int64, 18 | 0 | MISMATCH | MISMATCH | tgt=`hardtanh` trig=`remainder`; ref=[0.0, 0.0, 0.0] dev=[0.0, 1.0, 1.0]; near-equal to another live tensor |
| w53:531 | 0 | COMP | `ALIAS:L0` | **CLOBBER_NEAR:L0** | t6 | 2/2, 3/3 | - | float16, 16 | 1 | MISMATCH | MISMATCH | tgt=`erf` trig=`bitwise_left_shift`; ref=[0.0, 0.0, 0.0] dev=[0.0, 1.0, 1.0]; near-equal to another live tensor |
| w57:285 | 0 | COMP | `ALIAS:L0` | **CLOBBER_NEAR:L0** | t6 | 2/2, 3/3 | - | float32, 384 | 0 | MISMATCH | MISMATCH | tgt=`bitwise_left_shift` trig=`clamp;bitwise_left_shift`; ref=[0.0, 0.0, 0.0] dev=[0.0, 1.0, 1.0]; near-equal to another live tensor |
| w77:654 | 1 | COMP | `ALIAS:L0` | **CLOBBER_NEAR:L0** | t6 | 2/2, 3/3 | - | int64, 2 | 0 | MISMATCH | MISMATCH | tgt=`bitwise_or` trig=`bitwise_left_shift;logical_or`; ref=[0.0, 0.0] dev=[0.0, 1.0]; VACUOUS(n=2,matches=2); near-equal to another live tensor |
| w126:715 | 5 | COMP | `ALIAS:L4` | **CLOBBER_NEAR:L4** | t6 | 2/2, 3/3 | - | float32, 2 | 0 | MISMATCH | MISMATCH | tgt=`bitwise_xor` trig=`select_scatter`; ref=[0.0, 1.0] dev=[0.0, 2.0]; VACUOUS(n=2,matches=1); near-equal to another live tensor |
| w18:435 | 1 | COMP | `ALIAS:n0` | **CLOBBER_NEAR:n0** | t6 | 2/2, 3/3 | - | int64, 1 | 0 | MISMATCH | MISMATCH | tgt=`sign` trig=`bitwise_left_shift`; ref=[0.0] dev=[1.0]; VACUOUS(n=1,matches=3); near-equal to another live tensor |
| w63:150 | 2 | SIBL | `ALIAS:n0` | **CLOBBER_NEAR:n0** | t6 | 2/2, 3/3 | - | int32, 1 | 2 | OK | OK | tgt=`logical_not` trig=`amin`; ref=[0.0] dev=[1.0]; VACUOUS(n=1,matches=3); near-equal to another live tensor |
| w69:672 | 2 | COMP | `ALIAS:n0` | **CLOBBER_NEAR:n0** | t6 | 2/2, 3/3 | - | float32, 4 | 1 | MISMATCH | MISMATCH | tgt=`minimum` trig=`remainder`; ref=[-3.0, 0.0, 0.0] dev=[1.0, 0.0, 0.0]; VACUOUS(n=4,matches=5); near-equal to another live tensor |
| w94:418 | 0 | COMP | `ALIAS:n0` | **CLOBBER_NEAR:n0** | t6 | 2/2, 3/3 | - | float16, 1 | 2 | MISMATCH | MISMATCH | tgt=`unsqueeze_copy` trig=`_to_copy;remainder`; ref=[-6.0] dev=[1.0]; VACUOUS(n=1,matches=1); near-equal to another live tensor |
| w4:391 | 0 | COMP | `ALIAS:n1` | **CLOBBER_NEAR:n1** | t6 | 2/2, 3/3 | - | uint8, 8 | 2 | ERR 0x12 | ERR 0x12 | tgt=`ne` trig=`trunc;native_group_norm`; ref=[1.0, 0.0, 1.0] dev=[1.0, 1.0, 1.0]; near-equal to another live tensor |
| w58:331 | 0 | SIBL | `ALIAS:n10` | **CLOBBER_NEAR:n10** | t6 | 2/2, 3/3 | - | int64, 96 | 3 | OK | OK | tgt=`min` trig=`transpose_copy`; ref=[0.0, 0.0, 0.0] dev=[1.0, 1.0, 1.0]; near-equal to another live tensor |
| w113:165 | 0 | COMP | `ALIAS:n2` | **CLOBBER_NEAR:n2** | t6 | 2/2, 3/3 | - | uint8, 1 | 1 | MISMATCH | MISMATCH | tgt=`alias_copy` trig=`sum`; ref=[0.0] dev=[1.0]; VACUOUS(n=1,matches=1); near-equal to another live tensor |
| w43:252 | 3 | COMP | `ALIAS:n2` | **CLOBBER_NEAR:n2** | t6 | 2/2, 3/3 | - | float16, 1 | 5 | OK | OK | tgt=`squeeze_copy` trig=`floor_divide;exp`; ref=[2.7188] dev=[1.0]; VACUOUS(n=1,matches=1); near-equal to another live tensor |
| w8:209 | 0 | COMP | `ALIAS:n2` | **CLOBBER_NEAR:n2** | t6 | 2/2, 3/3 | - | uint8, 1 | 4 | MISMATCH | MISMATCH | tgt=`view_copy` trig=`bitwise_left_shift;gt;exp;amax`; ref=[0.0] dev=[1.0]; VACUOUS(n=1,matches=1); near-equal to another live tensor |
| w73:102 | 3 | COMP | `ALIAS:n3` | **CLOBBER_NEAR:n3** | t6 | 2/2, 3/3 | - | float16, 2 | 5 | OK | OK | tgt=`sin` trig=`floor_divide`; ref=[0.8413, 0.8413] dev=[0.0, 1.0]; VACUOUS(n=2,matches=1); near-equal to another live tensor |
| w70:398 | 2 | SIBL | `ALIAS:n2` | **NONFINITE** | t1b | 2/2, 3/3 | - | float16, 1 | 6 | OK | OK | tgt=`log10` trig=`squeeze_copy`; ref=[nan] dev=[-inf]; pos0 ref=nan dev=-inf |
| w70:615 | 2 | COMP | `ALIAS:n1` | **PARTIAL_ZERO** | t5 | 2/2, 3/3 | - | uint8, 4 | 0 | MISMATCH | MISMATCH | tgt=`eq` trig=`min;view_copy;bitwise_left_shift`; ref=[1.0, 1.0, 1.0] dev=[0.0, 0.0, 0.0]; 3/4 zeroed, rest match |
| w115:211 | 0 | SIBL | `SCALE:0.200` | **CLOBBER_EXACT:n1** | t6 | 2/2, 3/3 | - | float32, 4 | 4 | OK | OK | tgt=`mul` trig=`slice_scatter`; ref=[12.5875, 3.0141, 10.9693] dev=[2.5175, 0.6028, 2.1939]; VACUOUS(n=4,matches=2); byte-identical to another live tensor |
| w66:39 | 0 | SIBL | `SCALE:17.847` | **CLOBBER_EXACT:n6** | t6 | 2/2, 3/3 | - | float16, 2 | 3 | OK | OK | tgt=`hardtanh` trig=`fill`; ref=[-0.4482, -0.4482] dev=[-8.0, -8.0]; VACUOUS(n=2,matches=1); byte-identical to another live tensor |
| w25:146 | 0 | COMP | `SCALE:0.167` | **CLOBBER_NEAR:L0** | t6 | 2/2, 3/3 | - | int64, 16 | 0 | MISMATCH | MISMATCH | tgt=`abs` trig=`remainder`; ref=[0.0, 6.0, 6.0] dev=[0.0, 1.0, 1.0]; near-equal to another live tensor |
| w59:389 | 0 | COMP | `SCALE:-0.250` | **CLOBBER_NEAR:n0** | t6 | 2/2, 3/3 | - | float32, 2 | 1 | MISMATCH | MISMATCH | tgt=`maximum` trig=`remainder`; ref=[-4.0, -4.0] dev=[1.0, 1.0]; VACUOUS(n=2,matches=6); near-equal to another live tensor |
| w71:685 | 0 | SIBL | `SCALE:0.000` | **CLOBBER_NEAR:n0** | t6 | 2/2, 3/3 | - | bool, 384 | 3 | OK | OK | tgt=`logical_xor` trig=`split_with_sizes_copy`; ref=[1.0, 0.0, 1.0] dev=[0.0, 1.0, 0.0]; VACUOUS(n=384,matches=2); near-equal to another live tensor |
| w117:20 | 0 | COMP | `SCALE:-0.333` | **CLOBBER_NEAR:n1** | t6 | 2/2, 3/3 | - | float32, 2 | 4 | MISMATCH | MISMATCH | tgt=`mul` trig=`_to_copy;slice_copy;remainder`; ref=[-480.0, -480.0] dev=[160.0, 160.0]; VACUOUS(n=2,matches=1); near-equal to another live tensor |
| w103:698 | 0 | COMP | `SCALE:-7720456580759552.000` | **EXTREME_MAGNITUDE** | t2 | 2/2, 3/3 | - | int64, 3 | 0 | MISMATCH | MISMATCH | tgt=`gather` trig=`sub;bitwise_left_shift`; ref=[56.0, 56.0, 56.0] dev=[-4.323455642275676e+17, -4.323455642275676e+17, -4.323455642275676e+17]; |ref|max=56 |dev|max=4.323e+17 |
| w4:5 | 0 | COMP | `SCALE:1.500` | **INT_WRONG_VALUE** | t9 | 2/2, 3/3 | 0.5 / 0.5 | int16, 16 | 3 | OK | OK | tgt=`div` trig=`broadcast_to;_to_copy`; ref=[-2.0, -2.0, -2.0] dev=[-3.0, -3.0, -3.0]; max|delta|=1 |ref|max=2; ratio=1.5 consistent but VACUOUS (n_ratio=16, ref degenerate) |
| w62:425 | 0 | COMP | `SCALE:0.500` | **INT_WRONG_VALUE** | t9 | 2/2, 3/3 | 0.5 / 0.5 | int64, 4 | 0 | MISMATCH | MISMATCH | tgt=`abs` trig=`sum;broadcast_to;remainder;_to_copy`; ref=[6.0, 6.0, 6.0] dev=[3.0, 3.0, 3.0]; max|delta|=3 |ref|max=6; ratio=0.5 consistent but VACUOUS (n_ratio=4, ref degenerate) |
| w80:543 | 3 | COMP | `SCALE:1.083` | **INT_WRONG_VALUE** | t9 | 2/2, 3/3 | 0.1111 / 0.1111 | uint8, 96 | 2 | MISMATCH | MISMATCH | tgt=`bitwise_or` trig=`prod`; ref=[9.0, 9.0, 9.0] dev=[10.0, 10.0, 10.0]; max|delta|=1 |ref|max=9; ratio=1.083 consistent but VACUOUS (n_ratio=96, ref degenerate) |
| w105:478 | 4 | COMP | `SCALE:-6.945` | **NUMERIC** | t9 | 2/2, 3/3 | 4.825e+11 / 8 | float32, 4 | 3 | MISMATCH | MISMATCH | tgt=`remainder` trig=`select_scatter`; ref=[0.0, 1.0, 0.0] dev=[0.0, -7.0, -0.4825]; max|delta|=8 |ref|max=1; ratio=-6.945 consistent but VACUOUS (n_ratio=2) |
| w12:91 | 2 | COMP | `SCALE:0.711` | **NUMERIC** | t9 | 2/2, 3/3 | 0.2893 / 0.2893 | float32, 2 | 3 | ERR 0x12 | ERR 0x12 | tgt=`mul` trig=`native_group_norm;amin`; ref=[-1.6511, 1.176] dev=[-1.1735, 0.8359]; max|delta|=0.4776 |ref|max=1.651; ratio=0.7107 consistent but VACUOUS (n_ratio=2) |
| w19:324 | 1 | COMP | `SCALE:0.943` | **NUMERIC** | t9 | 2/2, 3/3 | 0.05738 / 0.05738 | float32, 384 | 3 | MISMATCH | MISMATCH | tgt=`maximum` trig=`broadcast_to;_to_copy;sum`; ref=[244.0, 244.0, 244.0] dev=[230.0, 230.0, 230.0]; max|delta|=14 |ref|max=244; ratio=0.9426 consistent but VACUOUS (n_ratio=384, ref degenerate) |
| w26:379 | 3 | COMP | `SCALE:3.078` | **NUMERIC** | t9 | 2/2, 3/3 | 2.078 / 2.078 | float16, 2 | 6 | OK | OK | tgt=`cos` trig=`index;squeeze_copy;elu`; ref=[-0.0597, -0.0597] dev=[-0.1837, -0.1837]; max|delta|=0.124 |ref|max=0.05969; ratio=3.078 consistent but VACUOUS (n_ratio=2, ref degenerate) |
| w37:595 | 0 | SIBL | `SCALE:0.250` | **NUMERIC** | t9 | 2/2, 3/3 | 0.75 / 0.75 | float32, 2 | 6 | OK | OK | tgt=`rsub` trig=`unbind_copy`; ref=[-8.0, -8.0] dev=[-2.0, -2.0]; max|delta|=6 |ref|max=8; ratio=0.25 consistent but VACUOUS (n_ratio=2, ref degenerate) |
| w3:423 | 0 | COMP | `SCALE:0.727` | **NUMERIC** | t9 | 2/2, 3/3 | 0.2729 / 0.2729 | float16, 4 | 5 | MISMATCH | MISMATCH | tgt=`div` trig=`sum`; ref=[0.0938, 0.0625, 0.125] dev=[0.0682, 0.0454, 0.0909]; max|delta|=0.03412 |ref|max=0.125; ratio=0.7271 consistent but VACUOUS (n_ratio=4) |
| w5:559 | 1 | COMP | `SCALE:1.325` | **NUMERIC** | t9 | 2/2, 3/3 | 0.3247 / 0.3247 | float16, 2 | 2 | OK | MISMATCH | tgt=`remainder` trig=`sigmoid`; ref=[0.0118, 0.0118] dev=[0.0156, 0.0156]; max|delta|=0.00383 |ref|max=0.0118; ratio=1.325 consistent but VACUOUS (n_ratio=2, ref degenerate) |
| w77:456 | 1 | COMP | `SCALE:2.000` | **NUMERIC** | t9 | 2/2, 3/3 | 1 / 1 | float16, 2 | 3 | OK | OK | tgt=`abs` trig=`atan2;remainder;lift_fresh_copy`; ref=[0.7852, 0.7852] dev=[1.5703, 1.5703]; max|delta|=0.7852 |ref|max=0.7852; ratio=2 consistent but VACUOUS (n_ratio=2, ref degenerate) |
| w98:638 | 1 | SIBL | `SCALE:1.438` | **NUMERIC** | t9 | 2/2, 3/3 | 0.4382 / 0.4382 | float32, 2 | 2 | OK | OK | tgt=`div` trig=`sum`; ref=[0.0869, 0.0869] dev=[0.125, 0.125]; max|delta|=0.03809 |ref|max=0.08691; ratio=1.438 consistent but VACUOUS (n_ratio=2, ref degenerate) |
| w29:743 | 1 | COMP | `SCALE:0.500` | **SCALE:0.5** | t8 | 2/2, 3/3 | - | float16, 64 | 5 | OK | OK | tgt=`div` trig=`_to_copy;broadcast_to`; ref=[-1.4795, 0.9194, -0.355] dev=[-0.7397, 0.4597, -0.1775]; n_ratio=64 std/|mean|=0 |
| w74:560 | 2 | COMP | `SCALE:0.750` | **SCALE:0.75** | t8 | 2/2, 3/3 | - | float32, 192 | 5 | OK | OK | tgt=`div` trig=`broadcast_to;_to_copy`; ref=[-1.7549, 0.8906, 4.8164] dev=[-1.3164, 0.668, 3.6133]; n_ratio=192 std/|mean|=0.000195 |
| w70:715 | 2 | SIBL | `SCALE:124.005` | **SCALE:124** | t8 | 2/2, 3/3 | - | float32, 144 | 3 | MISMATCH | OK | tgt=`maximum` trig=`remainder`; ref=[0.0, 0.0, 0.0] dev=[-0.9128, -1.7218, 0.0454]; n_ratio=16 std/|mean|=6.35e-08 |
| w114:621 | 2 | COMP | `SCALE:4221213.000` | **SCALE:4.221e+06** | t8 | 2/2, 3/3 | - | int32, 96 | 1 | OK | OK | tgt=`bitwise_right_shift` trig=`cumsum`; ref=[0.0, 0.0, 0.0] dev=[-4291676.0, 4202933.0, 4036894.0]; n_ratio=24 std/|mean|=0.00735 |
| w75:625 | 0 | COMP | `SCALE:-1.000` | **SHIFTED** | t7 | 2/2, 3/3 | - | float16, 2 | 2 | MISMATCH | MISMATCH | tgt=`native_layer_norm` trig=`exp;remainder`; ref=[-1.0, 1.0] dev=[1.0, -1.0]; same multiset, 2/2 positions differ shift+1 |
| w101:432 | 0 | COMP | `VALUE` | **EXTREME_MAGNITUDE** | t2 | 2/2, 3/3 | - | float32, 2 | 1 | MISMATCH | MISMATCH | tgt=`div` trig=`bitwise_left_shift`; ref=[0.0, -12.0] dev=[-1.7293822569102705e+18, -12.0]; |ref|max=12 |dev|max=1.729e+18 |
| w30:616 | 1 | COMP | `VALUE` | **EXTREME_MAGNITUDE** | t2 | 2/2, 3/3 | - | int64, 2 | 0 | MISMATCH | MISMATCH | tgt=`bitwise_xor` trig=`bitwise_left_shift`; ref=[0.0, 0.0] dev=[0.0, -9.223372036854776e+18]; |ref|max=0 |dev|max=9.223e+18 ==2^63 |
| w52:695 | 2 | COMP | `VALUE` | **EXTREME_MAGNITUDE** | t2 | 2/2, 3/3 | - | int64, 2 | 0 | MISMATCH | MISMATCH | tgt=`split_with_sizes_copy` trig=`bitwise_left_shift`; ref=[0.0, 0.0] dev=[7.205759403792794e+16, 0.0]; |ref|max=0 |dev|max=7.206e+16 ==2^56 |
| w5:342 | 0 | COMP | `VALUE` | **EXTREME_MAGNITUDE** | t2 | 2/2, 3/3 | - | int64, 128 | 0 | MISMATCH | MISMATCH | tgt=`pixel_unshuffle` trig=`bitwise_left_shift`; ref=[0.0, 0.0, 0.0] dev=[0.0, 2.305843009213694e+18, 2.305843009213694e+18]; |ref|max=0 |dev|max=2.306e+18 ==2^61 |
| w80:52 | 1 | COMP | `VALUE` | **EXTREME_MAGNITUDE** | t2 | 2/2, 3/3 | - | float32, 2 | 1 | MISMATCH | MISMATCH | tgt=`floor` trig=`bitwise_left_shift`; ref=[0.0, 0.0] dev=[0.0, -9.223372036854776e+18]; |ref|max=0 |dev|max=9.223e+18 ==2^63 |
| w89:35 | 0 | COMP | `VALUE` | **EXTREME_MAGNITUDE** | t2 | 2/2, 3/3 | - | int64, 2 | 0 | MISMATCH | MISMATCH | tgt=`t_copy` trig=`bitwise_left_shift`; ref=[0.0, 0.0] dev=[-4.323455642275676e+17, -4.323455642275676e+17]; |ref|max=0 |dev|max=4.323e+17 |
| w92:375 | 2 | COMP | `VALUE` | **EXTREME_MAGNITUDE** | t2 | 2/2, 3/3 | - | int64, 288 | 0 | MISMATCH | MISMATCH | tgt=`clamp` trig=`bitwise_left_shift;abs`; ref=[0.0, 0.0, 0.0] dev=[0.0, 1.152921504606847e+18, 1.152921504606847e+18]; |ref|max=0 |dev|max=1.153e+18 ==2^60 |
| w106:425 | 0 | COMP | `VALUE` | **INT_WRONG_VALUE** | t9 | 2/2, 3/3 | 41.5 / 41.5 | uint8, 2 | 0 | MISMATCH | MISMATCH | tgt=`masked_fill` trig=`sum`; ref=[0.0, 6.0] dev=[0.0, 255.0]; max|delta|=249 |ref|max=6 |
| w106:512 | 1 | COMP | `VALUE` | **INT_WRONG_VALUE** | t9 | 2/2, 3/3 | 4 / 1.333 | int64, 64 | 0 | MISMATCH | MISMATCH | tgt=`tril` trig=`remainder`; ref=[0.0, 2.0, 2.0] dev=[0.0, -1.0, -1.0]; max|delta|=4 |ref|max=3 |
| w32:411 | 0 | COMP | `VALUE` | **INT_WRONG_VALUE** | t9 | 2/2, 3/3 | 5 / 1.25 | int64, 36 | 0 | MISMATCH | MISMATCH | tgt=`diagonal_copy` trig=`ceil;remainder;sub`; ref=[2.0, 2.0, 4.0] dev=[-3.0, 2.0, 4.0]; max|delta|=5 |ref|max=4 |
| w35:212 | 3 | COMP | `VALUE` | **INT_WRONG_VALUE** | t9 | 2/2, 3/3 | 0.5 / 0.5 | uint8, 1 | 1 | MISMATCH | MISMATCH | tgt=`t_copy` trig=`sum`; ref=[2.0] dev=[1.0]; max|delta|=1 |ref|max=2 |
| w3:115 | 1 | SIBL | `VALUE` | **INT_WRONG_VALUE** | t9 | 2/2, 3/3 | 1 / 1 | int64, 1 | 3 | OK | OK | tgt=`sub` trig=`unbind_copy`; ref=[-3.0] dev=[-6.0]; max|delta|=3 |ref|max=3 |
| w70:750 | 3 | COMP | `VALUE` | **INT_WRONG_VALUE** | t9 | 2/2, 3/3 | 0.9922 / 0.9922 | int16, 1 | 4 | MISMATCH | MISMATCH | tgt=`sum` trig=`select_scatter`; ref=[258.0] dev=[2.0]; max|delta|=256 |ref|max=258 |
| w83:344 | 1 | COMP | `VALUE` | **INT_WRONG_VALUE** | t9 | 2/2, 3/3 | 1.6e+13 / 4 | int64, 12 | 0 | MISMATCH | MISMATCH | tgt=`bitwise_and` trig=`remainder`; ref=[3.0, 1.0, 0.0] dev=[3.0, 1.0, -4.0]; max|delta|=16 |ref|max=4 |
| w124:426 | 2 | SIBL | `VALUE` | **NONFINITE** | t1b | 2/2, 3/3 | - | float16, 1 | 7 | OK | OK | tgt=`min` trig=`atan2`; ref=[nan] dev=[-inf]; pos0 ref=nan dev=-inf |
| w127:717 | 0 | SIBL | `VALUE` | **NONFINITE** | t1b | 2/2, 3/3 | - | float32, 1 | 1 | MISMATCH | MISMATCH | tgt=`floor_divide` trig=`mul`; ref=[nan] dev=[inf]; pos0 ref=nan dev=inf |
| w39:585 | 0 | SIBL | `VALUE` | **NONFINITE** | t1b | 2/2, 3/3 | - | float32, 144 | 9 | OK | OK | tgt=`div` trig=`lt`; ref=[0.1608, 0.0402, -0.1608] dev=[0.0, 0.0, -0.0]; pos38 ref=inf dev=nan |
| w77:521 | 1 | SIBL | `VALUE` | **NONFINITE** | t1b | 2/2, 3/3 | - | float16, 2 | 7 | ERR 0x12 | ERR 0x12 | tgt=`atanh` trig=`unbind_copy`; ref=[nan, nan] dev=[inf, inf]; pos0 ref=nan dev=inf |
| w85:219 | 3 | SIBL | `VALUE` | **NONFINITE** | t1b | 2/2, 3/3 | - | float16, 2 | 4 | OK | SKIP | tgt=`log10` trig=`lift_fresh_copy`; ref=[nan, nan] dev=[-inf, -inf]; pos0 ref=nan dev=-inf |
| w26:67 | 0 | COMP | `VALUE` | **NUMERIC** | t9 | 2/2, 3/3 | 0.2166 / 0.2166 | float16, 4 | 4 | OK | OK | tgt=`sub` trig=`floor_divide`; ref=[4.6172, 4.6172, 4.6172] dev=[3.6172, 4.6172, 4.6172]; max|delta|=1 |ref|max=4.617 |
| w2:380 | 0 | COMP | `VALUE` | **NUMERIC** | t9 | 2/2, 3/3 | 0.04432 / 0.0362 | float16, 6 | 2 | ERR 0x12 | ERR 0x12 | tgt=`div` trig=`_log_softmax`; ref=[-0.0002, -2.1582, -1.2295] dev=[-0.0002, -2.0801, -1.2295]; max|delta|=0.07812 |ref|max=2.158 |
| w48:634 | 0 | COMP | `VALUE` | **NUMERIC** | t9 | 2/2, 3/3 | 0.1116 / 0.1116 | float16, 2 | 3 | OK | OK | tgt=`t_copy` trig=`asinh`; ref=[-3.377, 1.1104] dev=[-3.7539, 1.1104]; max|delta|=0.377 |ref|max=3.377 |
| w5:396 | 0 | COMP | `VALUE` | **NUMERIC** | t9 | 2/2, 3/3 | 0.316 / 0.316 | float16, 3 | 1 | MISMATCH | MISMATCH | tgt=`sigmoid` trig=`remainder;logical_not;bitwise_right_shif`; ref=[0.731, 0.731, 0.5] dev=[0.5, 0.731, 0.5]; max|delta|=0.231 |ref|max=0.731 |
| w61:506 | 3 | COMP | `VALUE` | **NUMERIC** | t9 | 2/2, 3/3 | 0.01558 / 0.01289 | float32, 6 | 6 | OK | OK | tgt=`tan` trig=`t_copy;native_group_norm;abs`; ref=[6.0586, 5.0156, 1.0869] dev=[6.0586, 5.0938, 1.0879]; max|delta|=0.07812 |ref|max=6.059 |
| w63:310 | 2 | SIBL | `VALUE` | **NUMERIC** | t9 | 2/2, 3/3 | 1.96 / 0.3984 | float16, 2 | 7 | OK | OK | tgt=`_softmax` trig=`max`; ref=[0.1689, 0.8311] dev=[0.5, 0.5]; max|delta|=0.3311 |ref|max=0.8311 |
| w69:691 | 2 | COMP | `VALUE` | **NUMERIC** | t9 | 2/2, 3/3 | 2 / 2 | float32, 2 | 2 | OK | OK | tgt=`atan2` trig=`elu`; ref=[-3.1406, 0.7852] dev=[3.1406, 0.7852]; max|delta|=6.281 |ref|max=3.141 |
| w87:603 | 3 | SIBL | `VALUE` | **NUMERIC** | t9 | 2/2, 3/3 | 0.0155 / 0.0155 | float16, 1 | 4 | ERR 0x12 | ERR 0x12 | tgt=`min` trig=`gt`; ref=[-64.5] dev=[-63.5]; max|delta|=1 |ref|max=64.5 |
| w92:427 | 3 | COMP | `VALUE` | **NUMERIC** | t9 | 2/2, 3/3 | 0.7529 / 0.7529 | float32, 1 | 2 | ERR 0x12 | ERR 0x12 | tgt=`min` trig=`native_group_norm`; ref=[-1.0] dev=[-1.7529]; max|delta|=0.7529 |ref|max=1 |
| w97:357 | 0 | COMP | `VALUE` | **NUMERIC** | t9 | 2/2, 3/3 | 11.35 / 0.3443 | float16, 64 | 3 | ERR 0x12 | ERR 0x12 | tgt=`unsqueeze_copy` trig=`native_group_norm`; ref=[-1.127, -0.1526, 1.293] dev=[-1.5039, -0.5684, 0.8203]; max|delta|=0.8582 |ref|max=2.492 |
| w98:671 | 0 | COMP | `VALUE` | **NUMERIC** | t9 | 2/2, 3/3 | 1.047 / 1.047 | float16, 8 | 2 | ERR 0x12 | ERR 0x12 | tgt=`native_group_norm` trig=`native_group_norm`; ref=[1.0, -1.0, -1.0] dev=[-0.0469, -1.3662, -0.0469]; max|delta|=1.047 |ref|max=1 |
| w99:467 | 1 | COMP | `VALUE` | **NUMERIC** | t9 | 2/2, 3/3 | 0.5662 / 0.5662 | float16, 2 | 5 | OK | OK | tgt=`asin` trig=`sign;floor_divide;_softmax`; ref=[0.5234, 0.5234] dev=[0.2727, 0.8198]; max|delta|=0.2964 |ref|max=0.5234 |
| w127:357 | 1 | SIBL | `VALUE` | **PARTIAL_ZERO** | t5 | 2/2, 3/3 | - | float16, 48 | 7 | OK | OK | tgt=`gt` trig=`transpose_copy`; ref=[0.0, 1.0, 1.0] dev=[0.0, 0.0, 1.0]; 14/48 zeroed, rest match |
| w85:286 | 1 | SIBL | `VALUE` | **PARTIAL_ZERO** | t5 | 2/2, 3/3 | - | int64, 64 | 3 | OK | SKIP | tgt=`ne` trig=`clone`; ref=[1.0, 1.0, 1.0] dev=[1.0, 1.0, 1.0]; 5/64 zeroed, rest match |
| w89:190 | 1 | SIBL | `VALUE` | **PARTIAL_ZERO** | t5 | 2/2, 3/3 | - | float32, 2 | 5 | OK | OK | tgt=`maximum` trig=`transpose_copy`; ref=[-0.0183, 0.8374] dev=[0.0, 0.8369]; 1/2 zeroed, rest match |
| w90:492 | 0 | COMP | `VALUE` | **PARTIAL_ZERO** | t5 | 2/2, 3/3 | - | float32, 3 | 3 | OK | OK | tgt=`squeeze_copy` trig=`pow`; ref=[1.0, 27.0, 1.0] dev=[0.0, 27.0, 0.0]; 2/3 zeroed, rest match |
| w11:82 | 0 | COMP | `VALUE` | **REF_ZERO_DEV_NONZERO** | t4b | 2/2, 3/3 | - | float16, 2 | 1 | MISMATCH | MISMATCH | tgt=`tan` trig=`split_with_sizes_copy;bitwise_left_shift`; ref=[0.0, 0.0] dev=[0.0, -84.75]; ref all~0, |dev|max=84.75 |
| w11:92 | 4 | COMP | `VALUE` | **REF_ZERO_DEV_NONZERO** | t4b | 2/2, 3/3 | - | float32, 6 | 2 | OK | OK | tgt=`repeat` trig=`remainder`; ref=[-0.0, 0.0, 0.0] dev=[-0.4825, 0.0, 0.0]; ref all~0, |dev|max=0.4825 |
| w13:255 | 1 | SIBL | `VALUE` | **REF_ZERO_DEV_NONZERO** | t4b | 2/2, 3/3 | - | float32, 1 | 3 | OK | OK | tgt=`unbind_copy` trig=`remainder`; ref=[0.0] dev=[1.1187]; ref all~0, |dev|max=1.119 |
| w17:612 | 0 | COMP | `VALUE` | **REF_ZERO_DEV_NONZERO** | t4b | 2/2, 3/3 | - | float32, 4 | 1 | MISMATCH | MISMATCH | tgt=`hardtanh` trig=`bitwise_left_shift`; ref=[0.0, 0.0, 0.0] dev=[0.0, 0.0, 0.0]; ref all~0, |dev|max=2 |
| w2:408 | 1 | COMP | `VALUE` | **REF_ZERO_DEV_NONZERO** | t4b | 2/2, 3/3 | - | float32, 18 | 0 | MISMATCH | MISMATCH | tgt=`sin` trig=`bitwise_left_shift`; ref=[0.0, 0.0, 0.0] dev=[0.0, -0.9999, -0.9999]; ref all~0, |dev|max=0.9999 |
| w55:621 | 3 | COMP | `VALUE` | **REF_ZERO_DEV_NONZERO** | t4b | 2/2, 3/3 | - | int32, 1 | 2 | MISMATCH | MISMATCH | tgt=`replication_pad3d` trig=`sum`; ref=[0.0] dev=[72.0]; ref all~0, |dev|max=72 |
| w85:444 | 3 | COMP | `VALUE` | **REF_ZERO_DEV_NONZERO** | t4b | 2/2, 3/3 | - | uint8, 4 | 0 | MISMATCH | MISMATCH | tgt=`bitwise_and` trig=`prod`; ref=[0.0, 0.0, 0.0] dev=[8.0, 0.0, 0.0]; ref all~0, |dev|max=8 |