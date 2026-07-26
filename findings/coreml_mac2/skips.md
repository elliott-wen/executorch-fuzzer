# CoreML — SKIP / CRASH tally (light; not the graph-opt focus)

Recorded for completeness — the focus of this run is the graph-optimization mismatches
([README.md](README.md)). These are **not** yet one-op-isolated per `analysis.md` steps 3–4; counts
+ verbatim reason classes only. Interim (feed ongoing).

## SKIP — device rejects a lowered graph (2,247 so far)
| count | verbatim reason class |
|---:|---|
| 1,032 | `Failed to load method forward, error: 0x:1 \| [ETCoreMLModel ...]` |
| 457 | `Caught an unknown exception!` |
| 260 | `method->execute() failed \| [tensor...]` |
| 192 | `method->execute() failed \| [coreml...]` |
| 177 | `Failed to load method forward \| [backend...]` |
| 80 | `method->execute() failed \| [op_fft...]` (fft ops — no CoreML/portable support) |
| 47 | `method->execute() failed \| [ETCoreML...]` |

Load-failures dominate: many multi-op graphs lower (partition) but the produced CoreML model fails to
**load** on device (`Failed to load method forward`) or throws an unknown exception at execute. To
turn these into filed coverage-gap bugs, run each rejected op as a one-op graph (workflow step 3/§SKIP)
and enumerate the forms — deferred (not the graph-opt focus).

## CRASH — native abort / load abort on device (221 so far)
Not bisected (per workflow, CRASH is localized by one-op isolation, not output-set ddmin). Deferred.

> Note vs `findings/coreml_mac` (prior run): that was a different corpus/run; this is corpus_v4 on
> the currently-connected Mac. No verdicts are borrowed across runs (analysis.md "no borrowed verdicts").
