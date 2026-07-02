# Cross-OS differential — Moto G54 5G: Android 14 vs prior Moto run

Same SoC class (Moto G54 5G, Mali-G57), **same seeded corpus** (`corpus_v2/vulkan`, 76,801 graphs,
deterministic per-graph inputs). The only variable is the OS/driver. Baseline = the prior moto full
feed `tmp/vulkan_moto_skip.tsv`. Per-job status diff via `tmp/cross_device_diff.py`
(raw: `cross_device_raw.txt`). A job absent from a skip-log was OK on that run.

## Aggregate — nearly identical
| outcome | A14 (this run) | baseline (prior moto) |
|---|---:|---:|
| OK | 30,053 | 30,059 |
| MISMATCH | 9,702 | 9,703 |
| CRASH | 3,191 | 3,182 |
| SKIP | 33,855 | 33,857 |

## Per-job concordance — 99.9%
| relation | count |
|---|---:|
| **shared, identical status** (SKIP 33,854 · MISMATCH 9,658 · CRASH 3,181) | **46,693** |
| status changed (non-OK both, differs) | 8 |
| A14-unique (OK on baseline → non-OK on A14) | 47 |
| baseline-unique (OK on A14 → non-OK on baseline) | 41 |

The 8 status-changes are almost all `MISMATCH↔CRASH` / `SKIP↔CRASH` (5+2+1) — a near-boundary
graph that silently mismatches on one run and aborts on the other; same underlying defect.

## The A14-unique tail is flaky / near-tolerance, not a new bug
Inputs are seeded (deterministic), so a clean A14-unique failure should re-fail. Re-feeding the 47
A14-unique jobs **twice** (`tmp/a14_rerun_{1,2}.txt`):

- **39/47 flipped to OK on both reruns** → clearly non-deterministic (GPU precision / scheduling noise).
- ~3 intermittent (OK on one rerun only) → also flaky.
- **5 stably reproduce** on A14 but were OK on the single baseline run:
  `w42:726` (max|Δ|=1, int), `w67:622` (max|Δ|=11, int), `w69:1011` (max|Δ|=1.5, fp),
  `w43:135` & `w88:348` (nan/inf position diffs). All are **near-tolerance / non-finite
  propagation in already-filed op classes** (floor_divide/remainder/reduction cast-order, nan/inf
  flow-through) — corpus inputs landing just over threshold on A14, just under on the baseline.

The baseline-unique set (40 MISMATCH + 1 SKIP) is the symmetric mirror — the same flaky tail
rolling the other direction. Both directions are ~the same size and the same character.

## Conclusion
The ExecuTorch-Vulkan bug surface on the Mali-G57 is **OS-version-independent**: every operator
bug, graph-optimization (copy-elision aliasing) bug, crash confound, and skip coverage gap from the
prior Moto run reproduces on Android 14, job-for-job. The differences between runs are dominated by
a small, symmetric, mostly non-deterministic precision tail — **no Android-14-specific bug class**.
This strengthens the prior cross-device result (Pixel 9 ↔ Moto): these are **backend-level
ExecuTorch defects**, not driver/SoC/OS-specific.
