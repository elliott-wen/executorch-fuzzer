# Workflow results — ASUS a12201 (Vulkan backend)

`analysis.md` run end-to-end on an ASUS a12201 against the full `corpus_v2/vulkan` corpus (76,801
lowered graphs). Every classification is **device-verified on this phone** — feed → full per-job
on-device bisection → single-op isolation with exact baked inputs.

- skip-log: `skip_log.tsv`  ·  per-job verdicts: `bisect_results.tsv`  ·  op enrichment: `op_enrichment.txt`
- per-operator catalog: `bugs/operators/INDEX.md`  ·  isolation provenance: `operator_isolation.tsv`
- cross-device: `cross_device_vs_moto_a14.txt`

## Triage (full corpus)
| outcome | count | share |
|---|---:|---:|
| OK | 31,227 | 40.7% |
| MISMATCH | 8,530 | 11.1% |
| CRASH | 3,187 | 4.2% |
| SKIP | 33,853 | 44.1% |
| TIMEOUT | 4 | 0.005% |

**The ASUS is more correct than the Moto G54:** 31,227 OK vs the Moto's 30,053 — **~1,170 fewer
mismatches** (8,530 vs 9,702), with SKIP and CRASH nearly identical. The GPU/driver differs, so a
band of fp16/precision-marginal graphs that the Moto fails come out OK here.

## Full per-job bisection — ALL 8,530 mismatches (coverage gate: PASS)
Every MISMATCH bisected on-device with the feeder's exact oracle (incl. input-mutation-output trim);
**8,530 verdict rows == 8,530 mismatch targets** asserted.

| verdict | count | share | meaning |
|---|---:|---:|---|
| **OPERATOR** | 7,025 | 82.4% | target diverges alone → kernel bug |
| **GRAPHOPT** | 903 | 10.6% | needs a co-returned sibling → copy-elision/aliasing |
| **NOREPRO** | 598 | 7.0% | didn't re-diverge → flaky / near-tolerance |
| **BISECT_CRASH** | 4 | 0.05% | re-processing crashes the host toolchain natively (`munmap`/SIGSEGV) — torch/executorch host bug, not a device bug |

## Per-operator bug catalog (single-op isolation, exact baked inputs)
195 OPERATOR-verdict candidate ops → **42 isolated GENUINE** (rest: 142 CLEAN/pass-alone, 8
LOWERFAIL, 3 NONFINITE_OUT) → after a **7× stability re-verification**, **24 confirmed**
(**21 deterministic operator bugs + 3 non-deterministic kernels**); 18 ruled out (reproduce eager on
re-run, or won't isolate as one op). Each confirmed op has `bugs/operators/<op>.md` + runnable
`repro_<op>.py` with **worst-element evidence**; see `bugs/operators/INDEX.md`. Far fewer than the
Moto's 49 — consistent with the ASUS's lower mismatch count.

## Graph-optimization bugs — CONFIRMED (copy-elision / aliasing)
The bisection labeled 903 mismatches GRAPHOPT. Unlike QNN (where strict partitioning made all such
verdicts unconfirmable artifacts), on Vulkan the targets **lower and run alone**, so the verdicts
are A/B-testable (`tmp/confirm_graphopt.py`: target alone must build READY + compare OK N×, and
target+sibling must diverge N×). Device-verified results (ASUS phone):

| target op | A/B result | class |
|---|---|---|
| **`expand_copy`** | **CONFIRMED** — OK alone, diverges with sibling, **4/4 instances** (w111:349, w117:568, w18:223, w21:331) | copy-elision / aliasing |
| **`clone`** | **CONFIRMED** — OK alone, diverges with sibling, 2/3 instances (w115:215, w122:117) | copy-elision / aliasing |
| `alias_copy`, `transpose_copy.int`, `clone` (w116:119) | flaky — does NOT re-diverge (near-tolerance / non-det) | not filed |
| `index_put` (903's #1 target, 65) | target alone → device **SKIP** at runtime (can't establish correct-alone) | inconclusive artifact |

**Confirmed graph-opt bug:** the **copy-elision / aliasing** class — `expand_copy` (reliably) and
`clone` are correct as the sole returned output but their result is **corrupted when a sibling
output is co-returned** (a buffer-planning / copy-elision interaction). This matches the Vulkan
copy-elision aliasing bug seen on the Moto, now independently device-confirmed on the ASUS. Caveat:
the raw 903 GRAPHOPT count overstates it — a sample shows the genuine bugs are the copy/view/alias
class; many other GRAPHOPT verdicts are flaky (don't re-diverge) or `index_put`-style runtime-SKIP
artifacts. Reproduce: `BISECT_BACKEND=vulkan BROKER_PORT=15554 python tmp/confirm_graphopt.py
w111:349 0 1`.

## Cross-device differential — ASUS a12201 vs Moto G54 (Android 14)
Same seeded corpus, per-job status diff (`cross_device_vs_moto_a14.txt`). Unlike the Moto-vs-Moto
run (99.9% concordant, same SoC), the ASUS is a **different GPU/driver**, so the device-unique tail
is much larger:

| relation | count |
|---|---:|
| **shared, identical status** (SKIP 33,850 · MISMATCH 7,611 · CRASH 3,180) | **44,641** |
| status changed | 14 |
| ASUS-unique (OK on Moto → non-OK on ASUS) | 919 (915 mismatch) |
| Moto-unique (OK on ASUS → non-OK on Moto) | 2,093 (2,087 mismatch) |

- **7,611 shared mismatches** = the **backend-level** ExecuTorch-Vulkan bugs that reproduce on both
  GPUs (the real, portable bug surface).
- **919 ASUS-unique + 2,093 Moto-unique** = **driver/precision-specific** divergences (many are
  non-finite nan/inf position diffs — different fp handling per GPU).
- **ASUS-specific signal:** `rsqrt.out` and `sqrt.out` jump up the ASUS mismatch ranking (a
  sqrt/rsqrt precision difference not prominent on the Moto's Mali-G57).

**Takeaway:** the operator/graph-opt/skip bug *classes* are the same across both devices (backend
defects), but the ASUS is quantitatively more correct, and the two GPUs disagree on a ~3,000-graph
precision-marginal band that is device-specific, not a backend bug.
