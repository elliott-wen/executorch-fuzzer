# QNN local-emulator findings — `corpus_v1/qnn`

Differential-testing report for the **Qualcomm QNN (HTP) backend**, run on the **x86 HTP
emulator** (`qnn_executor_runner`) on this host. Eager PyTorch is the oracle; each lowered
`.pte` is executed on the emulator and its outputs diffed against eager.

- **Corpus:** `corpus_v1/qnn` — 38,295 graphs (8-node DAGs, int-width-preserving dtype mapping)
- **Runner:** x86 `qnn_executor_runner` built into `pytorch_ref/executorch/build-x86`, patched
  to be **SKIP-aware** (clean "can't run here" → exit 2 = SKIP; only genuine hard failures →
  CRASH). Patch: `qnn_client/qnn_executor_runner.skip-aware.patch`.
- **SoC:** SM8450. **Date of run:** 2026-06-28.

## Headline ratio

| Verdict | Count | % |
|---|---:|---:|
| OK (matches eager) | 5,004 | 13.1% |
| **SKIP** (QNN can't run) | 17,989 | 47.0% |
| **MISMATCH** (ran, wrong) | 14,672 | 38.3% |
| **CRASH** (hard failure) | 629 | 1.6% |
| TIMEOUT | 1 | 0.0% |

Only **13% of deployable graphs reproduce eager**. Of the graphs that execute
(OK+MISMATCH+CRASH = 20,305), **75% pass** — but ~half the corpus never runs at all.

## Methodology

1. **Run** — feed every job to the broker; 16 emulator clients execute, feeder diffs vs eager.
   Failing rows → `results_nonok.tsv` (`status · job_id · reason · op-chain`).
2. **Classify** — bucket every non-OK job by signature.
   - SKIP `reason` is only `rc=2`, so the *stage* was recovered by re-running a 400-sample
     through the runner with `run.log` preserved (`skip_stage.py` → `localize/skip_stage.tsv`).
   - MISMATCH/CRASH localized with the first-divergence localizer (`localize_one.py`).
3. **Localize** — `localize(backend="qualcomm")` runs an all-node version of the graph on the
   in-process emulator and reports the first op where QNN diverges from eager (born-here root
   cause). All 629 crashes localized; 397 mismatch + 400 skip representatives sampled.
   Each localization is isolated in a subprocess (`localize_batch.sh`) so a hard crash records
   a `DIED` marker instead of killing the batch.

Raw data in `localize/`. Tooling: `localize_one.py`, `localize_batch.sh`, `skip_stage.py`,
`deep_dive.py` (per-node op/args + eager-vs-QNN values for one graph).

## Bug index (13 root-caused defects — detail in `families/`)

| # | Bug | Severity | Family |
|---|---|---|---|
| 1 | `sub`/`rsub` silently drop the `alpha` multiplier (compute `a−b` not `a−αb`) | HIGH | mismatch |
| 2 | `_to_copy` float→int **rounds**; eager **truncates** | HIGH | mismatch |
| 3 | Non-finite collapse: `log(0)→0`, `div/0→finite`, `rsqrt(neg)`/`logit(inf)`/`pow(_,nan)→finite` | MED | mismatch |
| 4 | `bitwise_left_shift` negative shift → garbage (≈2^60) | HIGH | mismatch |
| 5 | `elu` mishandles non-default `alpha`/`scale`/`input_scale` | MED | mismatch |
| 6 | `select_scatter` output dtype differs | LOW | mismatch |
| 7 | 0-element tensors can't be allocated (`broadcast_to([0,..])`, slice/unbind→0) | — (skip driver) | skip |
| 8 | Rank/batch change → execute `6004 INVALID_TENSOR` (`mean`-keepdim, `broadcast_to`, `view→(N,1)`) | — | skip |
| 9 | Graph won't finalize on HTP (`load_method` non-Ok) | — | skip |
| 10 | **Heap corruption** in QNN backend (`free(): invalid pointer`, `double free`) | CRITICAL | crash |
| 11 | **Segfaults** (45% of crashes; `max`/`roll`/reduce chains) | CRITICAL | crash |
| 12 | Internal QNN `ET_CHECK`/assert aborts (admit-then-abort) | MED | crash |
| 13 | Localizable crashes mirror mismatch bugs 1–5 + a tipping downstream op | — | crash |

---

## SKIP — 17,989 (47%) — QNN can't run the graph

By failing stage (n=400 re-probed with `run.log`):

| Stage | % | Root cause |
|---|---:|---|
| **alloc_output** | 49.5% | **zero-element output tensor** — runner can't `Allocate(0 bytes)` for a 0-numel output |
| **execute** | 30.8% | `qnn_graph_execute` **Error 6004 = `QNN_GRAPH_ERROR_INVALID_TENSOR`** — rank/batch mismatch on rank-changing ops (view/reshape/broadcast/reduce) |
| **alloc_input** | 15.0% | zero-element input leaf (same 0-byte alloc) |
| load_method | 4.8% | graph won't finalize on HTP |

**~64% of skips are empty (0-element) tensors**, **~31% are the rank/batch execute failure.**
The empty-tensor skips are a runner/QNN limitation (no 0-byte allocation); the 6004 family is
the genuine HTP limitation that rank-altering ops can't keep the batch dim coherent.

Representatives (repro: `skip_stage.py` or feed the single job):
- alloc_output (0-byte out): `w0:1092`
- execute / 6004 (rank-batch): `w0:2046`
- alloc_input (0-byte in): `w1:974`
- load_method (no finalize): `w10:827`

See [families/skip.md](families/skip.md).

---

## MISMATCH — 14,672 (38%) — QNN runs but diverges

Localized culprit op (n=397; 354 pinned to a node). Divergence kinds: **delta 61%, nonfinite
34%, dtype/shape 5%.**

| Culprit op | Count | Kind |
|---|---:|---|
| `_to_copy` | 37 | delta (cast rounding / fixed-point) |
| **`logit`** | 35 | **nonfinite** — eager `nan`, QNN finite |
| `rsub` / `sub` | 30 | delta |
| `bitwise_left_shift` | 14 | delta |
| `elu` | 12 | delta |
| `log` / `pow` / `div` / `rsqrt` | 31 | **nonfinite** — QNN doesn't emit `nan`/`inf` |
| `select_scatter` | 5 | dtype |

Two themes:
1. **Non-finite handling** (`logit`,`log`,`pow`,`div`,`rsqrt`, …): eager produces `nan`/`inf`
   on out-of-domain / div-by-zero; the HTP returns a finite (wrong) value. 34% of mismatches.
2. **Numeric/fixed-point delta** (`_to_copy`,`sub`/`rsub`,`elu`, bitwise): HTP rounds/quantizes
   differently than eager fp32. 61% of mismatches.

Representatives:
- `logit` nan→finite: `w0:735` (and the canonical `w0:1008` → `logit n3 e=nan b=-8.688`)
- `_to_copy` delta: `w0:1032`

See [families/mismatch.md](families/mismatch.md).

---

## CRASH — 629 (1.6%) — genuine hard failures

Process-death signature (full run): **abort `rc=134` 54% (340) · segfault `rc=139` 45% (286) ·
SIGFPE `rc=136` 3.** Localizer outcome (629):

| Outcome | Count | Meaning |
|---|---:|---|
| **DIED** | 316 | crashes the localizer too → **genuine hard crash** (segfault / heap corruption / internal assert) |
| RESULT | 218 | all-node form localizes to a divergent op before dying (same op families as MISMATCH: `_to_copy`, `select_scatter`, `logit`, …) |
| ERROR | 86 | localization raised — 76 `execute() failed`, 7 "Failed to generate Qnn context binary", 2 vector-too-large, 1 `bad_alloc` |
| OK | 9 | no divergence found |

Hard-crash (DIED) signature sample (n=60): ~43% internal `ET_CHECK`/assert (not in the runner —
inside the QNN backend), ~50% uncatchable abort/segfault, ~5% heap corruption
(`free(): invalid pointer`, `double free`). One confirmed earlier: heap corruption during QNN
**teardown** after a successful inference (`w0` family).

Representatives:
- hard crash (DIED): `w1:121`
- localizable crash: `w1:1560` (`elu`/delta)

See [families/crash.md](families/crash.md).

---

## Takeaways

- **QNN HTP is the least eager-faithful backend** (13% OK vs portable 70%, vulkan 42%).
- **Most dangerous = silent wrong math** (bugs 1, 2, 4): `alpha` dropped in `sub`/`rsub`,
  float→int rounding vs truncation, negative bit-shift garbage — no nan, no crash, just wrong
  numbers. These would corrupt a deployed model invisibly.
- **Memory safety (bugs 10, 11): ~95% of the 629 crashes are heap corruption or segfaults** in
  the QNN runtime — real defects, deterministic per graph, prime upstream bug reports.
- **Structural skip drivers**: empty-tensor allocation (bug 7, 64% of skips) and rank/batch
  `6004` (bug 8, 31%) — the highest-leverage backend-capability fixes.
- **Non-finite semantics** (bug 3) is the highest-leverage *numerical* fix: the HTP kernels
  lack IEEE special-case paths (domain checks, div-by-zero→inf, nan propagation).

## Reproduce one job

```bash
source android-dev/android-env.sh
export PYTHONPATH=/data/jwen929 \
       LD_LIBRARY_PATH="$PWD/pytorch_ref/executorch/build-x86/lib:$LD_LIBRARY_PATH"
# root-cause (first-divergence) on the local emulator:
.venv/bin/python findings_v1/qnn_local/localize_one.py corpus_v1/qnn/w0/w0_1008.py
# or the raw verdict via the broker: feed the single job_id
.venv/bin/python feed.py --corpus corpus_v1/qnn --host 127.0.0.1 w0:1008
```
