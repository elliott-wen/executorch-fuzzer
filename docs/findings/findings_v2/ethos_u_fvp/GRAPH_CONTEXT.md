# Ethos-U graph-optimization bugs — pinpointed mechanism

> Bug write-up with device-verified evidence + repro: **`bugs/graphopt-requant-scale.md`**.


The bisection produced 225 GRAPHOPT verdicts ("diverges only with a co-returned sibling"). To find
the actual bug I ran a **per-side A/B on device** for each: run the target's cone returning **only**
`[target]` (baseline), then `[target, sibling]` for each sibling, and classify. This filters two
confounds and names the real mechanism. **All device-verified on the Corstone-300 FVP.**

## First: two confounds removed
1. **int64 reference confound (the 2^63 red herring).** The huge `9.22e18 = 2^63` "garbage" deltas
   are **INT64 sentinels in the CPU reference**, not the device — they occur on int64/non-partitionable
   targets (`cumsum`, etc.) that run on the CPU int64 path. These mismatch **alone** (no sibling
   needed), so the A/B labels them **CONFOUND**, not graph-opt. (My first draft wrongly read these as
   device buffer-aliasing — corrected.)
2. **near-tolerance flakiness.** Many GRAPHOPT verdicts also mismatch alone or don't reproduce.

All 225 GRAPHOPT verdicts were A/B-tested; **62 gave a clean A/B** (163 were INCONCL — the target
won't partition to the NPU alone, a strict-partitioning limit):

| class | count | share of tested | meaning |
|---|---:|---:|---|
| **CONFOUND** | 36 | 58% | mismatches **alone** → not sibling-dependent (reference/quant/flaky), **not graph-opt** |
| **GRAPHOPT / DRIFT** | 21 | 34% | correct alone, wrong with a sibling, **bounded** wrong value |
| **GRAPHOPT / CLOBBER** | 5 | 8% | correct alone, wrong with a sibling, **non-finite / huge** wrong value |

So **~43% of testable GRAPHOPT verdicts are genuine sibling-dependent graph-opt bugs**; the majority
(57%) were confounds. The real graph-opt population is much smaller than the raw 225.

## The mechanism (deep-dive: `deep_graphopt.py` names each genuine case)
For each of the 19 genuine GRAPHOPT cases, I re-found the breaking sibling and characterized the wrong
device value. **Flakiness pervades even this set: 9/19 were NO_BREAK** — they did not re-mismatch with
any sibling on retest (non-deterministic sibling-dependence). Of the **10 that reproduced**:

| named mechanism | count | examples (device vs reference) | root cause |
|---|---:|---|---|
| **Requantization scale error** (`SCALE:r`) | 3 | `gelu` device≈ref×**−3.793**; `logit` ×**0.108**; `logit` ×**−1.5e18** | Vela requantizes the target with a **wrong output scale** — moderate, **sign-flipped**, or catastrophic — when a sibling forces a different partition/scale choice |
| **Dropped store** (`ZEROED`) | 2 | `view_copy`, `pixel_unshuffle` → device **all-zero**, ref non-zero | the target's output buffer is **never written** under the multi-output memory plan |
| **Non-finite** | 1 | `log1p` → device nan/inf, ref finite | overflow introduced by the sibling-dependent plan |
| **Unstructured** | 4 | `ge.Scalar`/`floor`/`sigmoid`/`logit` (0→1 flips, no simple relation) | comparison flips / mixed |
| (NO_BREAK) | 9 | `cos`, `masked_scatter`, `mul`, `min`, … | did not reproduce → flaky, not a stable graph-opt bug |

So the stably-reproducing, structurally-explained graph-opt bugs are a **small set** — dominated by
the **requantization scale error** (the only mechanism with a clean, repeatable numeric signature,
confirmed on `gelu`/`logit`) and a secondary **dropped store**. Everything else is unstructured or
flaky. This is much narrower than the raw 225 GRAPHOPT verdicts implied.

Co-returning a second output
changes how Vela partitions and assigns `(scale, zero_point)` across the graph, and the target's
output is then dequantized with the **wrong scale** — sometimes just off (×0.108), sometimes
sign-flipped (×−3.79), sometimes wild (×−1.5e18). It is a **quantization-parameter / partitioning
bug in the Vela compiler**, surfacing only when the multi-output plan differs from the single-output
one. A secondary mechanism is a **dropped store** (the buffer isn't written → zeros).

## Contrast with QNN
QNN's graph-opt bug was **memory-planning buffer aliasing** (a sibling's live buffer verbatim-clobbers
the target — `../qnn_emulator/bugs/graphopt-fill-buffer-aliasing.md`). Ethos-U is **different**: the
dominant mechanism is a **requant-scale error**, with only a minority looking clobber-like (ZEROED /
non-finite). Same corpus, structurally different graph-opt bug per backend.

## Honest limits
- 163/225 GRAPHOPT cases are **INCONCL** (target won't lower alone) — not testable by this A/B on a
  strict-partitioning NPU. The 43%-genuine / 57%-confound split is over the **testable** 62.
- The mechanism table is over the genuine cases the deep-dive resolved; a few genuine cases are
  UNSTRUCTURED / NO_BREAK (non-deterministic). Raw data: `pinpoint/`, `deepdive/` (staged as
  `graphopt_pinpoint.tsv`, `graphopt_mechanism.tsv`).
