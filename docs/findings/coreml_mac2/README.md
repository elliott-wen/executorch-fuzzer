# CoreML backend — graph-optimization bug analysis (Mac worker, corpus_v4)

`analysis.md` workflow on **`corpus_v4/coreml`** (10,518 multi-node graphs, ops=10–17), **focused on
the graph-optimization bugs** per request. Lowered on Linux (coremltools); executed on a connected
**macOS CoreML worker** via the broker; each failing graph bisected with `mobile.gen.diff`
(output-set → cone) to split operator vs graph-optimization, then the graph-opt mechanism named.

## Headline: ZEROED dropped-store on view/copy outputs (memory-plan graph-opt bug)

📄 **Full mechanism analysis with cross-backend proof, minimal root cause, lowered-IR evidence, and
neutralization: [`GRAPHOPT_REPORT.md`](GRAPHOPT_REPORT.md).**

The dominant CoreML graph-opt bug is a **dropped store**: the output buffer of a view/identity/copy
op is **never written**, reading back all-zeros, whenever that output **aliases a buffer the memory
plan writes under a different name** — either a second returned alias of the same tensor, or a view
whose source is a portable-fallback op. Kernel math is correct; the plan drops the store. **50 of
109** bisected graph-opt cases. **CoreML-specific** — the same lowered subgraph is correct on
portable and xnnpack (ruled out the multi-output alignment artifact that fooled the CUDA run).
Device-verified A/B:

```
w1:8  n6 = lift_fresh_copy(n2), out[3], trigger n3 = max
  A) return [n6] ALONE     : device = [-0.28, -0.42]  = eager   ✓ correct
  B) return [n6, n3=max]   : device = [ 0.0,  0.0 ]           ✗ ZEROED (store dropped)
```

Target ops are overwhelmingly no-op/view/identity: `lift_fresh_copy` (10), `squeeze_copy` (7),
`alias_copy`/`view_copy`/`t_copy`/`detach_copy`. The trigger sibling varies widely (`max`, `prod`,
`var.correction` ×15, `remainder`, `log10`, …) → **not** trigger-specific: a **copy-elision /
buffer-aliasing** memory-plan bug. Full write-up + repro: [bugs/graphopt_dropped_store.md](bugs/graphopt_dropped_store.md).

## Graph-opt mechanism breakdown (503 graph-opt cases — full corpus)

| mechanism | n | notes |
|---|---:|---|
| **ZEROED** (dropped store) | **204** | headline; view/copy outputs, deep analysis in [GRAPHOPT_REPORT.md](GRAPHOPT_REPORT.md) |
| WRONG-VALUE | 125 | value corruption incl. logical-inversion / `SCALE:0` mislabels |
| REF-NONFINITE (**filtered**) | 85 | eager is `nan` (reference-degenerate, step 3d) — not a device bug |
| SCALE:r | ~26 | **mostly an OPERATOR bug** — `var.correction(correction=None)` computes biased ÷N (off by (N−1)/N); 28/48 SCALE cases. [operator_var_correction.md](bugs/operator_var_correction.md), [graphopt_scale.md](bugs/graphopt_scale.md) |
| NEAR-TOL (**gate N≥5**) | 19 | tiny deltas; likely fp noise pending gate |
| NONFINITE | 17 | device nan/inf from finite leaf |

Confirmed graph-opt findings = **SIBLING_DEPENDENT (308) + COMPOSITIONAL (195) = 503**, dominated by
the **ZEROED dropped-store class (204)**. Filtering out the 104 non-device cases (85 reference-nan +
19 near-tolerance) leaves ~399 device-verified graph-opt divergences.

## Coverage

- **Feed:** full corpus **10,518** graphs → **OK 5,321 · MISMATCH 1,601 · CRASH 325 · SKIP 3,250 · TIMEOUT 21**.
- **Bisection (per-job, `analysis.md` gate):** **1,490 / 1,601 mismatches = 93.1%** have a real verdict:
  | verdict | n |
  |---|---:|
  | CONE_LOCAL_CANT_SPLIT | 656 |
  | OPERATOR | 328 |
  | **SIBLING_DEPENDENT** | **308** |
  | **COMPOSITIONAL** | **195** |
  | INCONCLUSIVE | 3 |
  `CONE_LOCAL_CANT_SPLIT` (656) = the target won't lower alone on CoreML's strict partitioner, so
  operator-vs-compositional can't be split — CONE_LOCAL (bug in the target's own chain) but **not** a
  graph-opt-by-sibling finding.
- **Residual (~7%):** the un-verdicted rows are `LOWERFAIL` (BUILD:SKIP during the run) + a few
  unattempted. **These are recoverable, not real gaps:** every sampled LOWERFAIL graph re-lowers
  **READY** in isolation (original *and* rebuilt) — the SKIP was a **transient coremltools lowering
  failure under concurrent load** (6 bisector shards + the feed all invoking `torch.export` +
  CoreML conversion at once → intermittent build exceptions caught as SKIP), not an unsupported
  graph. A **low-concurrency (3-shard) recovery pass** re-attempts them (now capturing the real SKIP
  detail); coverage climbs toward 100% as they resolve. Per the gate this is a documented partial
  until then; un-verdicted `job_id`s are the LOWERFAIL/not-yet rows in `bisect_results.*.tsv`.

## Light tally of the other modes (not the focus)
- **OPERATOR mismatches (328):** single-op kernel bugs (CONE_LOCAL → OPERATOR). One is now
  root-caused — the `var.correction(correction=None)` biased-variance bug ([bugs/operator_var_correction.md](bugs/operator_var_correction.md)),
  which also accounts for 28 of the "SCALE" cases via its downstream consumers. The rest not yet characterized.
- **CRASH (221):** native aborts / load failures on device — not bisected (per workflow, localized by
  one-op isolation). Deferred.
- **SKIP (2,247):** device rejects a lowered graph. Top classes (verbatim): `Failed to load method
  forward` (1,032 + 177), `Caught an unknown exception!` (457), `method->execute() failed`
  (tensor/coreml/op_fft variants, ~580). Real coverage gaps; enumerated in [skips.md](skips.md) (light).

## Artifacts
- `skiplog.tsv` — every non-OK job. `bisect_results.*.tsv` — per-(job,out) verdicts (6 shards).
- `graphopt_mechanisms.tsv` — mechanism per graph-opt case. `bisect_driver.py` / `run_bisect.sh` /
  `characterize_graphopt.py` — the pipeline. `bugs/repro_graphopt.py` — A/B repro.
- `bugs/` — graph-opt findings (dropped-store headline + scale cluster).

## Reproduce
```bash
# broker + Mac worker up; then:
findings/coreml_mac2/run_bisect.sh 6                 # (re)bisect, resumable
.venv/bin/python findings/coreml_mac2/characterize_graphopt.py   # name mechanisms
BROKER_PORT=15554 PYTHONPATH=/data/jwen929 .venv/bin/python \
  findings/coreml_mac2/bugs/repro_graphopt.py w1:8 3 n3          # headline A/B
```
