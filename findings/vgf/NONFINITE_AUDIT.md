# The VGF `NONFINITE` bucket (194 rows) and the 105 unlabelled rows — audited by measurement

`GRAPHOPT_REPORT.md` reports **984 VGF graph-optimization mismatches** from `graphopt_all.tsv`
and counts **194 `NONFINITE`** rows plus **105 rows with no mechanism at all** (76 blank, 29 `?`)
inside that total. This audit re-measured **every one of the 194** rows on four execution paths
and adjudicated the 105.

**Headline: 34 of 194 (17.5%) survive as VGF-specific graph-level non-finite findings, and even
those 34 are structurally the whole-output-replacement signature, not a numeric mechanism.
`NONFINITE` should not be reported as a graph-optimization category. 60 rows are provably not
VGF findings, 65 are the already-reported ZEROED dropped-store bug mislabelled (63 of them on the
`SIBLING_DEPENDENT` axis that `ZEROED_AUDIT.md` independently confirms), 15 are input perturbation,
20 are unfalsifiable. Separately, the 29 `?` rows are confirmed non-reproductions
and must leave the 984; the 76 blank rows are *unmeasured*, not unreal — 11/12 sampled reproduce,
mostly as `ZEROED`.**

Probes, raw data and the full 194-row table: `findings/vgf/nonfinite_probe/`.

---

## 0. Environment and commands (all results below are reproducible)

```bash
export PYTHONPATH=/data/jwen929
export CUDA_VISIBLE_DEVICES=
export MODEL_CONVERTER_LIB_DIR=/opt/nvidia/nsight-systems/2024.6.2/host-linux-x64
export PATH=/data/jwen929/mobile/.venv/bin:$PATH
cd findings/vgf/nonfinite_probe

# 0. confirm the vgf backend actually probes available (else every build_job returns SKIP)
python -c "from mobile.gen.export import get_backend; print(get_backend('vgf'))"
#   -> <mobile.gen.export.backends.vgf.VgfBackend object at 0x...>   [verified]

# 1. VGF A/B replay + non-finite signatures + leaf-input check, 32-row stride sample, N=3
for s in 0 1 2 3; do python -u nf_vgf.py --out nf_s$s.jsonl --limit 32 --nshards 4 --shard $s & done

# 2. the SAME reconstructed B-side graphs on portable / xnnpack / openvino,
#    one subprocess per (backend,row), MOBILE_BACKENDS-isolated
for b in portable xnnpack openvino; do bash nf_xb.sh nf_all.jsonl $b xb_$b.jsonl & done

# 3. population sweep: A-side AND B-side of ALL 194 rows on portable and on xnnpack
for s in 0..7; do MOBILE_BACKENDS=portable python -u nf_full.py portable fp_s$s.jsonl --shard $s --nshards 8 & done
for s in 0..7; do MOBILE_BACKENDS=xnnpack  python -u nf_full.py xnnpack  fx_s$s.jsonl --shard $s --nshards 8 & done

# 4. VGF delegated_ops for the B-side graph of ALL 194 rows (build-only)
for s in 0..5; do python -u nf_vd.py vd_s$s.jsonl --shard $s --nshards 6 & done

# 5. VGF device VALUES for ALL 194 rows (non-finite signature, all-zero flag, fp16-limit flag)
for s in 0..7; do python -u nf_vgfall.py va_s$s.jsonl --shard $s --nshards 8 & done

# 6. confound tests
python -u nf_anc.py     anc_s0.jsonl --only <20 COMPOSITIONAL survivors>   # live-ancestor drift
python -u nf_operand.py opd_s0.jsonl --only <34 survivors>                 # target-operand drift
python -u nf_bits.py w1:238 w70:246 w113:116 ...                           # raw device bit patterns

# 7. adjudication
python -u nf_final.py        # cross-backend verdict over all 194
python -u nf_adjudicate.py   # final categories over all 194
```

Coverage actually achieved: **194/194 rows measured on VGF (device values), 194/194 on portable,
194/194 on xnnpack, 194/194 for VGF delegation**, plus a 32-row sample with openvino added and
N=3 A/B replay.

---

## 1. First, an arithmetic correction

`graphopt_all.tsv` has **no header row**: `wc -l` = 984 = 984 *data* rows. The `ab` column
partitions them exactly:

| `ab` | rows | what the harness recorded |
|---|--:|---|
| `AB_OK` | 872 | A correct, B diverged ≥1/3 |
| `DEV:TIMEOUT` | 76 | **exception path** — nothing was measured |
| `B_NOREPRO` | 29 | B never diverged in 3 tries |
| `A_ALSO_DIVERGES` | 7 | excluded by the report already |

`NONFINITE` is **194** rows (the earlier count of 193 came from skipping line 1 as a header).

---

## 2. What the 105 unlabelled rows are (task 3)

### 2.1 The writer, read from the source

`tmp/vgf_go_deep.py` emits a blank mechanism from exactly **one** place — the exception handler:

```python
except Exception as e:
    fh.write(f"{job}\t{out_idx}\t{v}\tERR\t\t{str(e)[:40]}\t0\t{NREP}\t\n")
```

so a blank `mechanism` means the row is `tgt_op=ERR`, `ab=<exception text>`, `n_div=0`. All 76 carry
`DEV:TIMEOUT`, raised by `send()` when the broker/device did not answer inside
`ISO_DEV_TIMEOUT` (**default 40 s**). **Nothing about these rows was observed.**

`?` is the initial value of `mm`, only ever overwritten inside `if diverges_at(...)`:

```python
ndiv=0; mm="?"
for k in range(NREP):
    if diverges_at(...): ndiv+=1; mm = mech(...) if mm=="?" else mm
```

so `mechanism == "?"` is definitionally `ndiv == 0` — B **did not diverge**, which is also why all
29 carry `ab=B_NOREPRO`.

### 2.2 Re-measured on the local VGF runner (no broker, no 40 s cap)

12 of each sampled (24 rows), A/B rebuilt identically, N=3:

| bucket | sampled | A correct | B reproduces 3/3 | B never diverges | recovered mechanism |
|---|--:|--:|--:|--:|---|
| blank / `DEV:TIMEOUT` (76) | 12 | 12/12 | **11/12** | 1/12 | 7 `ZEROED`, 3 `NONFINITE`, 1 `VALUE` |
| `?` / `B_NOREPRO` (29) | 12 | 12/12 | **0/12** | **12/12** | — |

Real target ops recovered for the blank rows (the TSV stores `ERR` in that column):

| job | out | real target op | re-measured mechanism | n_div |
|---|---|---|---|--:|
| w0:45 | 0 | `fmod` | ZEROED | 3/3 |
| w100:505 | 0 | `clone` | VALUE | 3/3 |
| w101:39 | 1 | `slice_copy` | ZEROED | 3/3 |
| w102:563 | 0 | `hardtanh` | NONFINITE | 3/3 |
| w103:492 | 0 | `constant_pad_nd` | ZEROED | 3/3 |
| w106:109 | 0 | `detach_copy` | NONFINITE | 3/3 |
| w109:223 | 0 | `cumsum` | — (OK ×3) | 0/3 |
| w113:301 | 0 | `tan` | ZEROED | 3/3 |
| w116:365 | 1 | `view_copy` | ZEROED | 3/3 |
| w124:164 | 2 | `expm1` | ZEROED | 3/3 |
| w13:599 | 2 | `erf` | NONFINITE | 3/3 |
| w39:443 | 0 | `acos` | ZEROED | 3/3 |

**Verdict on the 105.**
- The **29 `?` rows do not belong in the 984**. They are non-reproductions by the harness's own
  gate and by re-measurement (0/12 reproduce). Remove them.
- The **76 blank rows do belong, but not as unclassified filler**: they are a *measurement*
  failure (a 40 s device timeout), not a non-finding. 11/12 reproduce deterministically, and the
  mechanism they recover is predominantly **`ZEROED`** — so they reinforce the report's headline
  dropped-store bug and should be re-run and re-labelled, not silently carried as blanks.
- Consequence: the 984 total is wrong in both directions and its mechanism table is incomplete for
  8% of its own rows.

> Note: **`w102:563` — the single case `GRAPHOPT_REPORT.md` §3.2b cites to discount all non-finite
> findings — is one of these blank rows.** It is not in the `NONFINITE` bucket at all.

---

## 3. The classifier's actual behaviour on non-finite values (task 4)

`tmp/vgf_go_deep.py::mech()` labels from the **finiteness mask only**:

```python
rf = torch.isfinite(r); df = torch.isfinite(d)
if (rf != df).any(): return "NONFINITE"
if d.abs().max() < 1e-6 and r.abs().max() > 1e-3: return "ZEROED"
...
```

whereas the *detector*, `net/compare.py::_cmp`, tests `isnan()` / `== inf` / `== -inf`
position-for-position. So the information exists at detection time and is **thrown away at
labelling time**. Measured (unit matrix, `mech()` copied verbatim):

| case | `_cmp` verdict | `mech()` label |
|---|---|---|
| finite → `nan` (device invents nan) | MISMATCH | `NONFINITE` |
| finite → `+inf` | MISMATCH | `NONFINITE` |
| `nan` → finite (saturation) | MISMATCH | `NONFINITE` |
| `+inf` → `65504` (fp16 limit) | MISMATCH | `NONFINITE` |
| `-inf` → `-65504` | MISMATCH | `NONFINITE` |
| `nan` → `+inf` | MISMATCH | **`VALUE`** |
| `+inf` → `nan` | MISMATCH | **`VALUE`** |
| `-inf` → `+inf` (**sign flip of infinity**) | MISMATCH | **`VALUE`** |
| `+inf` → `-inf` (**sign flip of infinity**) | MISMATCH | **`VALUE`** |
| all `-inf` → all `+inf` | MISMATCH | **`VALUE`** |
| finite `1e30` → `65504` (**fp16 saturation of a FINITE value**) | MISMATCH | **`VALUE`** |
| finite `1e-8` → `0` (fp16 underflow) | OK | (n/a) |
| `nan` positions move, same count | MISMATCH | `NONFINITE` |

Concretely:

1. **It tests `isfinite`, never `isnan`.** It therefore cannot distinguish `nan` from `inf`, nor
   `+inf` from `-inf`. A `-inf → +inf` flip and a `nan → inf` substitution are labelled `VALUE`.
   The `NONFINITE` label carries **zero** direction information, and the 332-row `VALUE` bucket is
   contaminated with genuine non-finite divergences.
2. **A finite value saturating to the fp16 limit ±65504 is not detected** as a format effect — it
   is `VALUE` (or `SCALE`), indistinguishable from generic noise. Conversely an `inf → 65504`
   *is* labelled `NONFINITE`, so the same format effect lands in two different buckets depending
   on which side is non-finite.
3. **`NONFINITE` pre-empts `ZEROED` and `ALIAS`.** The tests are ordered, so a device output that
   is **all zeros** — the report's headline dropped-store signature — is labelled `NONFINITE`
   whenever the *reference* happens to contain a nan or inf:

   ```
   mech(ref=[nan,1,2], dev=[0,0,0]) -> NONFINITE      # a dropped store, mislabelled
   mech(ref=[3,  1,2], dev=[0,0,0]) -> ZEROED
   mech(ref=[nan,1,2], dev=[7,8,9]) -> NONFINITE      # an ALIAS clobber, mislabelled
   ```
   This is not hypothetical: **§6 shows 65 of the 194 rows are exactly this case.** The report's
   `ZEROED` count is therefore an *under*-count and `NONFINITE` an over-count. (See also
   `findings/vgf/ZEROED_AUDIT.md`, which independently finds the SIBLING_DEPENDENT ZEROED symptom
   real and VGF-specific — the axis these 63 mislabelled rows belong to.)
4. It also mislabels the neighbouring bucket in the other direction, as the task's spot check
   found: a real non-finite divergence with matching finiteness masks lands in `VALUE`.

---

## 4. NONFINITE: is it a graph-opt finding at all? (task 2)

Three prerequisites, checked on all 194:

**(a) Leaf inputs. Not an injection artifact — negative result, cleanly.**
`gen/graph/ir.py::_inject` does exist and 28022/28878 `corpus_v4/vgf` job files call it, but
`corpus_v4` sets **`_NONFIN_P = 0.0005`**, not 5%. Evaluating each graph's seeded leaves:
**0 / 194** rows have any non-finite leaf input (`Counter({'no': 194})`). Every nan/inf in these
graphs is produced by op semantics (`log(0)`, `div` by 0, fp16 overflow), not injected. So the
"reference artifact" hypothesis is **refuted**.

**(b) A side. Correct — the recorded `AB_OK` holds.** On the 32-row N=3 sample, the VGF A side
(target alone / baked cone) is `OK` **32/32**. On the full 194 on portable, the A side is `OK`
183/194 (7 MISMATCH, 4 cannot run). So the target is not simply a broken kernel.

**(c) Determinism. Holds — unlike SHAPE.** All 194 rows were recorded at `n_div=3/3`; there is
no `n_div=1/3` row anywhere in the bucket (contrast `shape_probe/README.md`, where the 4 rows
that evaporated were exactly the 4 recorded at 1/3). Re-measured: **194/194 reproduce MISMATCH**
on the local runner, and 32/32 at 3/3 with the mechanism label replaying as `NONFINITE` in
193/194 (1 replayed as `VALUE`). *Determinism is not what is wrong with this bucket.*

**But the graph-level attribution fails for other reasons.**

**(d) 12 rows have no VGF delegate at all.** `build_job(srcB,'vgf').delegated_ops == 0` for 12 of
the 194 reconstructed B-side graphs. The graph that "diverged on VGF" contains zero VGF-delegated
ops — it ran entirely on portable kernels. These cannot be VGF findings by construction. (This is
the same delegation-metric inflation recorded elsewhere in this repo: `ops>0` in the job file is
not the same as the diverging output being inside a delegate.)

**(e) 46 rows diverge with *no delegate whatsoever*.** Running the identical reconstructed B-side
graph on `portable` (`delegated_ops == 0` by construction) reproduces the divergence in **58/194**
rows; excluding the 12 above, **46** are backend-attributable-to-nothing. In the 32-row detail
sample the portable device values are frequently *bit-identical* to VGF's:

| job | target | eager ref | VGF device | portable device |
|---|---|---|---|---|
| w70:246 | `hardtanh` | 16× `-4.0` | 16× `nan` | 16× `nan` |
| w108:539 | `pow` | 128× `1.0` | 72× `+inf` | 72× `+inf` |
| w119:234 | `acos` | 12× `1.5703` | 7× `nan` | 7× `nan` |
| w95:30 | `sqrt` | 4× `0.0` | 3× `+inf` | 3× `+inf` |
| w59:83 | `cumsum` | 4× `0.0` | 3× `-inf` | 3× `-inf` |
| w62:771 | `slice_copy` | `1994.0` | `+inf` | `+inf` |
| w91:426 | `div` | 2× `nan` | 2× `0.0` | 2× `0.0` |

`nf_bits.py` confirms these are ordinary IEEE encodings (`ffc00000` = −qNaN, `7f800000` = +inf),
not garbage: the CPU computes the same thing. **These are eager-vs-ExecuTorch reference-pipeline
divergences** (export/decomposition changes the numerics of a singular op), present with no
backend involved. 39 of the 46 have a clean portable *A* side, so they are even
"graph-context-dependent" — just not *backend* graph-context-dependent.

**(f) 15 more are input perturbation, not graph optimization.** For a `COMPOSITIONAL` row the A
side bakes the ancestors as fp32 **constants** while the B side runs them **live**, so B feeds the
target a *different input*. `nf_anc.py` emits the live ancestors alongside the target: for the 20
`COMPOSITIONAL` rows that had survived every other filter, **17/20 have a live ancestor that
already diverges from eager**. The target op set is `div`×7, `floor_divide`×3, `rsqrt`, `atanh`,
`_softmax`, `prod`, `glu` — functions that are singular or catastrophically amplifying near their
domain boundary. The non-finite output is downstream error propagation. Only 3/20 have every live
ancestor correct.

---

## 5. NONFINITE, cross-backend: the VGF-only vs shared split (task 1)

### 5.1 The report's note does not generalize — and its own exemplar is atypical

`GRAPHOPT_REPORT.md` §3.2b: *"on the non-finite case `w102:563` (ref = nan), xnnpack and openvino
also diverge (they saturate nan → −1.0) — i.e. NONFINITE/saturation divergences are shared across
delegates, not VGF-specific."*

Reproduced exactly, and the observation is correct **for that case**:

| backend | w102:563 out0, ref `[nan, 0.484]` |
|---|---|
| portable | `[nan, 0.484]` → **OK** |
| xnnpack | `[-1.0, 0.484]` → MISMATCH |
| openvino | `[-1.0, 0.484]` → MISMATCH |

But `-1.0` is `hardtanh`'s clamp minimum: xnnpack and openvino clamp nan to the lower bound where
eager propagates it. That is a **hardtanh-specific op-semantics difference in those two
delegates**, not a law about delegate number handling. And **`w102:563` is not a `NONFINITE` row**
— it is one of the 76 blank `DEV:TIMEOUT` rows (§2). The note generalizes from a single case
outside the bucket it is used to discount.

### 5.2 Measured on the bucket itself, all 194 rows

| control | result over 194 rows |
|---|--:|
| `portable` B side: MISMATCH | **58** |
| `portable` B side: OK | 116 |
| `portable` B side: cannot execute | 20 |
| `xnnpack` B side: MISMATCH | 60 |
| `xnnpack` B side: OK | 107 |
| `xnnpack` B side: cannot execute | 27 |
| VGF `delegated_ops == 0` | 12 |

Cross-backend verdict:

| verdict | rows | % |
|---|--:|--:|
| no VGF delegate in the diverging graph | 12 | 6.2% |
| **also diverges on portable (no delegate at all)** | **46** | **23.7%** |
| neither portable nor xnnpack can execute — unfalsifiable | 20 | 10.3% |
| shared with another delegate (xnnpack diverges too) | 2 | 1.0% |
| VGF-only candidate | 114 | 58.8% |

Of the 114 VGF-only candidates, **93 have a genuine second-delegate control** (xnnpack absorbed
≥1 op into a delegate *and* was correct); the other 21 have `xnnpack delegated_ops == 0`, i.e.
xnnpack ≡ portable there and there is no independent delegate control. **56 of 194** rows are
xnnpack-non-delegating, which is a real limit on that control.

The 32-row detail sample with openvino added is in
`nonfinite_probe/xbackend_32sample.jsonl` (openvino: 12 MISMATCH, 10 OK, 5 partitioner SKIP,
5 execution error — a weaker control than the report implies).

**So the report's note is refuted as a general claim** (only 2/194 are shared with another
delegate), **but the finding it was used to justify is refuted more strongly**: the divergences
that are not VGF-specific are shared with *portable*, i.e. with no delegate at all.

---

## 6. Final adjudication of all 194 rows

Categories are applied in order; every row was measured on every axis.

| # | category | rows | % | why it is not a VGF graph-opt non-finite finding |
|---|---|--:|--:|---|
| A | no VGF delegate in the diverging graph | 12 | 6.2% | `delegated_ops == 0`; nothing VGF ran |
| B | diverges on `portable` too | 46 | 23.7% | reproduces with no delegate; reference-pipeline divergence |
| C | no CPU control (portable **and** xnnpack cannot execute) | 20 | 10.3% | unfalsifiable with available controls |
| D | shared with another delegate | 2 | 1.0% | not VGF-specific |
| E | **device output is entirely ZERO** | **65** | **33.5%** | this is the already-reported `ZEROED` dropped store; `mech()`'s `isfinite`-first ordering mislabelled it because the *reference* held a nan/inf |
| F | `COMPOSITIONAL` with an already-wrong live ancestor | 15 | 7.7% | input perturbation into a singular op, not graph optimization |
| H | **survives** | **34** | **17.5%** | VGF-only, reproducible, A-side clean, device value not all-zero |

By original verdict axis — the `COMPOSITIONAL` half collapses almost completely:

| | `SIBLING_DEPENDENT` (108) | `COMPOSITIONAL` (86) |
|---|--:|--:|
| A no VGF delegate | 0 | 12 |
| B portable too | 6 | 40 |
| C unfalsifiable | 6 | 14 |
| D shared | 2 | 0 |
| E is ZEROED | 63 | 2 |
| F ancestor wrong | 0 | 15 |
| **H survives** | **31** | **3** |

### 6.1 Two more negative results

- **No fp16 saturation anywhere.** `dev_sat65504` is `False` for **194/194**: not one device value
  in the whole bucket is `±65504`. The report's framing of non-finite divergence as
  "TOSA/delegate saturation" and "the live cone overflows fp16" is **unsupported by any observed
  device value**. (105/194 outputs *are* `float16`, so the range is genuinely narrow — the
  saturation just never happens; VGF produces `inf`/`nan`/`0`, not the clamped limit.)
- **The `NONFINITE` label survives replay but means the wrong thing.** 193/194 replay as
  `NONFINITE` under `mech()`, and **73/194 device outputs are entirely zero** (65 of those 73
  survive filters A–D and become category E; the other 8 were already excluded for a stronger
  reason). The label is *stable*; it is simply not a mechanism.

### 6.2 Even the 34 survivors are the whole-output-replacement signature

| survivor property | count |
|---|--:|
| `SIBLING_DEPENDENT` / `COMPOSITIONAL` | 31 / 3 |
| with a real second-delegate control (xnnpack deleg > 0, OK) | 26 / 34 |
| **device output is a single uniform non-finite constant** | **25 / 34** (17/26 on tensors with > 1 element) |
| target op in the `log / log2 / log10` family | 13 / 34 |
| target op singular at 0 or domain-restricted (`log*`, `reciprocal`, `rsqrt`, `fmod`, `remainder`, `pow`, `*_norm`, `acos(h)`) | 26 / 34 |

The pattern is consistent and diagnostic:

| job | target | eager reference | VGF device |
|---|---|---|---|
| w111:267 | `log10` | 68 `-inf` + 60 `0.0` | **all 128 `-inf`** |
| w43:395 | `log` | 7 `-inf`, finite max 3.55 | **all 16 `-inf`** |
| w47:580 | `log10` | 3 `-inf` + 3 `0.0` | **all 6 `-inf`** |
| w24:275 | `log2` | 3 finite (max 2.81) | **all 3 `-inf`** |
| w27:126 | `reciprocal` | `0.5` | `+inf` |
| w37:260 | `fmod` | 144 finite (max 1.75) | **all 144 `nan`** |
| w50:276 | `native_group_norm` | 12 finite (max 1.72) | **all 12 `nan`** |

`log(0) = -inf`, `1/0 = +inf`, `x % 0 = nan`, zero-variance normalisation `= nan`. A **uniform**
non-finite constant across an entire output is the signature of the operand or the output buffer
being replaced wholesale — the same `ZEROED` family observed one op downstream — not of a
per-element numeric mechanism.

**This could not be proven directly.** `nf_operand.py` tries to emit the target's own operand
alongside the target; **31 of 34 builds returned `SKIP`** (adding one output makes the VGF
partitioner reject the graph — itself a notable observability limit). The 3 usable cases are all
`COMPOSITIONAL`, and in all 3 the operand is *within tolerance* but not bit-identical
(w52:88 `pow`: operand ref `1.0` vs device `1.00195`, output goes `nan`) — knife-edge
amplification again. So the whole-output-replacement reading of the 34 is **strongly indicated by
the value structure but not established**; treat the 34 as *at most* 34.

---

## 7. Verdict

**How many of the 194 `NONFINITE` rows are defensible as VGF graph-level findings: at most 34
(17.5%), and none of them are defensible as a distinct *non-finite* mechanism.**

- **60 rows (31%) are provably not VGF findings**: 12 have no VGF delegate, 46 reproduce on
  portable with no delegate at all, 2 are shared with xnnpack.
- **65 rows (33.5%) are the `ZEROED` dropped-store bug already reported in §3**, mislabelled
  because `mech()` tests `isfinite` before all-zero. **63 of the 65 are `SIBLING_DEPENDENT`** —
  precisely the axis the parallel `ZEROED_AUDIT.md` finds defensible (243 reproducing rows) — so
  they should be moved there, taking the dropped-store count to **~306**. The 2 `COMPOSITIONAL`
  ones inherit that audit's caveat that COMPOSITIONAL ZEROED rows are mostly upstream single-op
  bugs. Net effect: the mislabelling *understated* the report's strongest finding while inflating
  its weakest category.
- **15 rows (7.7%)** are input-perturbation amplification through a singular op.
- **20 rows (10.3%)** cannot be adjudicated: no CPU control can execute the graph.
- **34 rows (17.5%)** survive every control. 25 of the 34 have a *uniform* non-finite output —
  the same whole-output-replacement signature, seen through `log`/`reciprocal`/`fmod`.

### 7.1 Recommended edits to `GRAPHOPT_REPORT.md`

1. **Delete `NONFINITE` as a mechanism category** (§2 table, §4 "`NONFINITE` (86, the live cone
   overflows fp16 and the plan introduces nan/inf)"). Redistribute: 65 → `ZEROED`, 60 → excluded,
   15 → excluded, 20 → unresolved, 34 → fold into the dropped-store/`ZEROED` discussion as
   "observed through a singular op", flagged as not independently confirmed.
2. **Remove the fp16-saturation narrative for non-finite values** — 0/194 device values reach
   ±65504. (§4's `VALUE` fp16 claim also deserves re-examination: the corpus lowers with
   `TOSA-1.0+FP+INT`, i.e. *float*, and 89/194 of these outputs are already `float32`.)
3. **Fix the §3.2b note.** Its exemplar `w102:563` is a `DEV:TIMEOUT` row, not a `NONFINITE` row,
   and its conclusion is contradicted by the bucket (2/194 shared with another delegate).
4. **Recompute the total.** Drop the 29 `B_NOREPRO` rows (984 → 955); re-run and re-label the 76
   `DEV:TIMEOUT` rows instead of carrying them blank (they are mostly `ZEROED`).
5. **Add `portable` as a mandatory control to the harness.** 58/194 rows in this bucket alone
   reproduce with no delegate; a no-delegate control run costs one extra lowering and would have
   removed a quarter of the category before any device time was spent.

### 7.2 Recommended classifier fixes

- Order the tests **all-zero → alias → non-finite → scale → value**, not non-finite first.
- Classify non-finite divergence by *direction and kind*, from `_cmp`'s existing
  `isnan` / `==inf` / `==-inf` masks: `NAN_INVENTED`, `INF_INVENTED`, `NAN_LOST`, `INF_SIGN_FLIP`,
  `NAN_TO_INF`, and separately `FORMAT_SATURATION` for `|dev| == 65504` (either side).
- Add a **uniformity** flag: a device output that is one constant value is a
  whole-output-replacement, and should never be reported as a numeric mechanism.
- For `COMPOSITIONAL` rows, emit the **live ancestors** and gate the verdict on the ancestors
  being correct — otherwise the axis measures input perturbation, not graph optimization.
- Do not write `mechanism=""` for an exception; write `mechanism=UNMEASURED` and exclude such
  rows from every headline count.

---

## 8. Appendix A — the 34 survivors

| job | out | verdict | target op | vgf/xnn deleg | out dtype | ref non-finite | device kind | portable | xnnpack |
|---|---|---|---|---|---|---|---|---|---|
| w106:252 | 1 | COMP | `pow` | 2/0 | float32 | no | dev_nonfinite | OK | OK |
| w111:267 | 1 | SIBL | `log10` | 8/4 | float32 | yes | ref_nonfinite,dev_nonfinite | OK | OK |
| w125:56 | 1 | SIBL | `cos` | 5/2 | float32 | yes | ref_nonfinite | OK | OK |
| w127:292 | 2 | SIBL | `gelu` | 7/0 | float16 | no | dev_nonfinite | OK | OK |
| w21:434 | 0 | SIBL | `log` | 2/1 | float16 | no | dev_nonfinite | OK | OK |
| w24:275 | 3 | SIBL | `log2` | 3/1 | float16 | no | dev_nonfinite | OK | OK |
| w27:126 | 1 | SIBL | `reciprocal` | 6/3 | float16 | no | dev_nonfinite | OK | OK |
| w31:158 | 2 | SIBL | `acosh` | 5/2 | float16 | yes | ref_nonfinite,dev_nonfinite | OK | OK |
| w36:416 | 0 | SIBL | `acos` | 6/2 | float16 | yes | ref_nonfinite | OK | OK |
| w37:260 | 0 | SIBL | `fmod` | 5/3 | float32 | no | dev_nonfinite | OK | OK |
| w39:122 | 1 | SIBL | `cos` | 6/4 | float16 | no | dev_nonfinite | OK | OK |
| w40:44 | 4 | COMP | `native_layer_norm` | 5/2 | float32 | yes | ref_nonfinite | OK | OK |
| w41:314 | 0 | SIBL | `pow` | 5/None | float32 | yes | ref_nonfinite,dev_nonfinite | OK | CANT_RUN |
| w43:395 | 1 | SIBL | `log` | 5/3 | float16 | yes | ref_nonfinite,dev_nonfinite | OK | OK |
| w47:580 | 0 | SIBL | `log10` | 3/1 | float32 | yes | ref_nonfinite,dev_nonfinite | OK | OK |
| w4:230 | 2 | SIBL | `log2` | 3/None | float32 | yes | ref_nonfinite,dev_nonfinite | OK | CANT_RUN |
| w50:276 | 1 | SIBL | `native_group_norm` | 3/0 | float32 | no | dev_nonfinite | OK | OK |
| w52:88 | 0 | COMP | `pow` | 2/0 | float32 | no | dev_nonfinite | OK | OK |
| w55:44 | 0 | SIBL | `asinh` | 5/2 | float16 | yes | ref_nonfinite,dev_nonfinite | OK | OK |
| w59:736 | 0 | SIBL | `log` | 6/2 | float16 | no | dev_nonfinite | OK | OK |
| w63:609 | 2 | SIBL | `fmod` | 5/2 | float32 | yes | ref_nonfinite,dev_nonfinite | OK | OK |
| w66:723 | 1 | SIBL | `log10` | 3/None | float16 | no | dev_nonfinite | OK | CANT_RUN |
| w67:687 | 1 | SIBL | `log10` | 3/1 | float16 | no | dev_nonfinite | OK | OK |
| w68:192 | 2 | SIBL | `log2` | 5/1 | float16 | no | dev_nonfinite | OK | OK |
| w6:114 | 1 | SIBL | `expand_copy` | 5/2 | float32 | yes | ref_nonfinite | OK | OK |
| w73:147 | 1 | SIBL | `remainder` | 4/1 | float32 | yes | ref_nonfinite,dev_nonfinite | OK | OK |
| w84:144 | 0 | SIBL | `log` | 4/1 | float16 | no | dev_nonfinite | OK | OK |
| w89:380 | 2 | SIBL | `rsqrt` | 5/4 | float16 | no | dev_nonfinite | OK | OK |
| w89:496 | 3 | SIBL | `elu` | 4/0 | float32 | yes | ref_nonfinite,dev_nonfinite | OK | OK |
| w89:779 | 0 | SIBL | `clamp` | 4/2 | float32 | yes | ref_nonfinite | OK | OK |
| w91:350 | 1 | SIBL | `log2` | 2/1 | float32 | yes | ref_nonfinite,dev_nonfinite | OK | OK |
| w95:4 | 0 | SIBL | `rsqrt` | 9/5 | float16 | no | dev_nonfinite | OK | OK |
| w95:533 | 0 | SIBL | `masked_scatter` | 4/3 | float32 | yes | ref_nonfinite | OK | OK |
| w97:358 | 0 | SIBL | `log2` | 5/2 | float16 | no | dev_nonfinite | OK | OK |

## 9. Appendix B — the 32-row cross-backend detail sample (openvino included)

`vd` = verdict axis; `deleg_ops` = ops absorbed into each backend's delegate for this exact
graph. A backend with `deleg_ops == 0` is not an independent delegate control. `DIVERGES` =
MISMATCH vs the eager reference; `SKIP` = partitioner rejected; `ERR` = runtime could not execute.

| job | out | vd | target op | out dtype | vgf/xnn/ov deleg_ops | A alone | VGF B n_div | leaf finite? | direction | portable | xnnpack | openvino | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w0:163 | 0 | COMP | `select_scatter` | float32 | 0/0/8 | OK | 3/3 | yes | dev_invents | DIVERGES | DIVERGES | DIVERGES | NO_VGF_DELEGATE |
| w108:539 | 0 | COMP | `pow` | float32 | 1/0/6 | OK | 3/3 | yes | dev_invents | DIVERGES | DIVERGES | DIVERGES | REF_PIPELINE (portable too) |
| w113:116 | 0 | COMP | `tanh` | float32 | 2/1/2 | OK | 3/3 | yes | dev_invents | ERR | ERR | DIVERGES | PORTABLE_CANT_RUN (inconclusive) |
| w119:234 | 0 | COMP | `acos` | float16 | 1/0/2 | OK | 3/3 | yes | dev_invents | DIVERGES | DIVERGES | DIVERGES | REF_PIPELINE (portable too) |
| w123:362 | 2 | COMP | `clone` | float16 | 1/1/6 | OK | 3/3 | yes | dev_invents | DIVERGES | DIVERGES | OK | REF_PIPELINE (portable too) |
| w127:292 | 2 | SIBL | `gelu` | float16 | 7/0/8 | OK | 3/3 | yes | dev_invents | OK | OK | OK | VGF_ONLY |
| w1:238 | 0 | COMP | `ceil` | float16 | 2/1/2 | OK | 3/3 | yes | dev_invents | ERR | ERR | DIVERGES | PORTABLE_CANT_RUN (inconclusive) |
| w23:771 | 1 | SIBL | `select_copy` | float32 | 8/2/0 | OK | 3/3 | yes | ref_saturated | OK | OK | SKIP | VGF_ONLY |
| w25:174 | 1 | SIBL | `expm1` | float32 | 6/1/4 | OK | 3/3 | yes | ref_saturated | OK | OK | OK | VGF_ONLY |
| w30:359 | 2 | COMP | `floor_divide` | float32 | 4/0/5 | OK | 3/3 | yes | ref_saturated | OK | OK | OK | VGF_ONLY |
| w35:9 | 1 | SIBL | `prod` | float16 | 2/4/14 | OK | 3/3 | yes | ref_saturated | OK | OK | ERR | VGF_ONLY |
| w38:488 | 3 | COMP | `min` | float32 | 0/0/2 | OK | 3/3 | yes | dev_invents | DIVERGES | DIVERGES | ERR | NO_VGF_DELEGATE |
| w42:13 | 1 | COMP | `div` | float16 | 2/1/3 | OK | 3/3 | yes | dev_invents | OK | OK | ERR | VGF_ONLY |
| w43:395 | 1 | SIBL | `log` | float16 | 5/3/6 | OK | 3/3 | yes | both | OK | OK | ERR | VGF_ONLY |
| w48:107 | 0 | COMP | `div` | float16 | 5/0/5 | OK | 3/3 | yes | both | OK | OK | OK | VGF_ONLY |
| w4:689 | 2 | COMP | `log2` | float32 | 0/0/1 | OK | 3/3 | yes | both | DIVERGES | DIVERGES | DIVERGES | NO_VGF_DELEGATE |
| w52:88 | 0 | COMP | `pow` | float32 | 2/0/3 | OK | 3/3 | yes | dev_invents | OK | OK | DIVERGES | SHARED_WITH_DELEGATE |
| w56:670 | 2 | SIBL | `trunc` | float16 | 5/1/0 | OK | 3/3 | yes | ref_saturated | OK | OK | SKIP | VGF_ONLY |
| w59:83 | 1 | COMP | `cumsum` | float16 | 0/0/1 | OK | 3/3 | yes | dev_invents | DIVERGES | DIVERGES | DIVERGES | NO_VGF_DELEGATE |
| w61:424 | 0 | COMP | `atanh` | float16 | 1/0/2 | OK | 3/3 | yes | dev_invents | DIVERGES | DIVERGES | DIVERGES | REF_PIPELINE (portable too) |
| w62:771 | 2 | COMP | `slice_copy` | float16 | 1/0/1 | OK | 3/3 | yes | dev_invents | DIVERGES | DIVERGES | DIVERGES | REF_PIPELINE (portable too) |
| w68:139 | 1 | SIBL | `logit` | float16 | 4/6/26 | OK | 3/3 | yes | ref_saturated | OK | OK | OK | VGF_ONLY |
| w6:114 | 1 | SIBL | `expand_copy` | float32 | 5/2/5 | OK | 3/3 | yes | ref_saturated | OK | OK | OK | VGF_ONLY |
| w70:246 | 0 | COMP | `hardtanh` | float32 | 3/1/4 | OK | 3/3 | yes | dev_invents | DIVERGES | DIVERGES | DIVERGES | REF_PIPELINE (portable too) |
| w73:147 | 1 | SIBL | `remainder` | float32 | 4/1/0 | OK | 3/3 | yes | both | OK | OK | SKIP | VGF_ONLY |
| w79:400 | 0 | COMP | `remainder` | float32 | 3/0/3 | OK | 3/3 | yes | dev_invents | ERR | ERR | OK | PORTABLE_CANT_RUN (inconclusive) |
| w83:586 | 0 | SIBL | `slice_copy` | float16 | 5/2/0 | OK | 3/3 | yes | ref_saturated | OK | OK | SKIP | VGF_ONLY |
| w87:247 | 1 | SIBL | `transpose_copy` | float16 | 9/2/9 | OK | 3/3 | yes | ref_saturated | OK | OK | OK | VGF_ONLY |
| w89:496 | 3 | SIBL | `elu` | float32 | 4/0/0 | OK | 3/3 | yes | both | OK | OK | SKIP | VGF_ONLY |
| w91:426 | 3 | COMP | `div` | float32 | 6/1/6 | OK | 3/3 | yes | ref_saturated | DIVERGES | DIVERGES | DIVERGES | REF_PIPELINE (portable too) |
| w95:30 | 0 | COMP | `sqrt` | float16 | 1/0/3 | OK | 3/3 | yes | dev_invents | DIVERGES | DIVERGES | OK | REF_PIPELINE (portable too) |
| w97:358 | 0 | SIBL | `log2` | float16 | 5/2/6 | OK | 3/3 | yes | dev_invents | OK | OK | ERR | VGF_ONLY |

The 32-row sample verdicts: 16 VGF_ONLY, 8 also-diverges-on-portable, 4 no-VGF-delegate,
3 portable-cannot-run, 1 shared-with-openvino. Direction split: 17 "device invents non-finite"
(of which 10 reproduce on portable), 10 "reference was non-finite, device returned finite"
(9 of which are VGF-only), 5 both. A-side OK 32/32; recorded `n_div` 3/3 for all 32 and
reproduced 3/3 for all 32; leaf inputs finite 32/32.

---

## 10. Data files

| file | contents |
|---|---|
| `nonfinite_probe/adjudication_194.md` | the full 194-row table (job, out, verdict, target, VGF deleg, ref non-finite, dev all-zero, portable, xnnpack, category) |
| `nonfinite_probe/vgf_bside_194.jsonl` | VGF device values + non-finite signature + all-zero/fp16-limit flags, all 194 |
| `nonfinite_probe/portable_194.jsonl` | portable A-side and B-side, all 194 |
| `nonfinite_probe/xnnpack_194.jsonl` | xnnpack A-side and B-side, all 194 |
| `nonfinite_probe/vgf_deleg_194.jsonl` | VGF `delegated_ops` for the B-side graph, all 194 |
| `nonfinite_probe/vgf_ab_32sample.jsonl` | N=3 VGF A/B replay + leaf-input check, 32-row sample |
| `nonfinite_probe/xbackend_32sample.jsonl` | portable / xnnpack / openvino on the 32-row sample |
| `nonfinite_probe/ancestor_drift.jsonl` | live-ancestor divergence for the 20 COMPOSITIONAL survivors |
| `nonfinite_probe/operand_probe.jsonl` | target-operand probe on the 34 survivors (31 build-SKIP) |
| `nonfinite_probe/ub_blank.jsonl`, `ub_q.jsonl` | re-measurement of the 76 blank / 29 `?` rows |
| `nonfinite_probe/adjudicated.json` | machine-readable final categories |
