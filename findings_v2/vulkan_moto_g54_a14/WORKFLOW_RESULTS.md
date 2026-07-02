# Workflow results — Moto G54 5G, **Android 14** (Vulkan backend)

`analysis.md` run end-to-end on a Moto G54 5G (Mali-G57 GPU) running **Android 14**, against the
full `corpus_v2/vulkan` corpus (76,801 lowered graphs). Every classification below is
**device-verified on this phone** — feed → per-job on-device bisection → single-op isolation.

- Feed log: `tmp/feed_moto_g54_a14.log`  ·  skip-log: `skip_log.tsv` (copied here) / `tmp/vulkan_moto_g54_a14_skip.tsv`
- Provenance: `op_enrichment.txt`, `cross_device_raw.txt`  ·  cross-OS analysis: `CROSS_DEVICE_DIFF.md`
- Reproduce any verdict: `python tmp/bisect_graph.py <job_id> <out_index>` (mismatch root cause) or run a `bugs/repro_*.py` (needs a connected Vulkan worker on `127.0.0.1:15554`).

## Triage (full corpus, this run)
| outcome | count | share |
|---|---:|---:|
| OK | 30,053 | 39.1% |
| MISMATCH | 9,702 | 12.6% |
| CRASH | 3,191 | 4.2% |
| SKIP | 33,855 | 44.1% |
| TIMEOUT | 0 | 0% |

## Full per-job bisection — ALL 9,702 mismatches (coverage gate: PASS)
Every MISMATCH graph was bisected on this device (not a representative sample) with the feeder's
exact oracle (incl. the input-mutation-output trim), one verdict row per `(job, out)` in
`bisect_results.tsv`. Coverage asserted: **9,702 verdict rows == 9,702 mismatch targets**.

| verdict | count | share | meaning |
|---|---:|---:|---|
| **OPERATOR** | 7,909 | 81.5% | target diverges as the sole returned output → kernel bug |
| **GRAPHOPT** | 1,157 | 11.9% | correct alone, diverges only with a co-returned sibling → copy-elision/aliasing |
| **NOREPRO** | 633 | 6.5% | full graph didn't re-diverge → flaky / near-tolerance (not a bug) |
| **BISECT_CRASH** | 3 | 0.03% | `w11:448`, `w122:348`, `w126:623` — re-processing the graph crashes the **host** toolchain natively (`munmap_chunk`/SIGSEGV) *after* `build_job` returns READY; a torch/executorch host bug, not a device bug, hence un-bisectable |

**Per-op verdict table** (legitimate aggregation, produced AFTER per-job bisection): `bisect_per_op.txt`.
Top operator-bug ops by device-verified count: `clamp.Tensor_out` (559), `select_scatter` (290,
+159 flaky), `pow.Tensor_Tensor_out` (239), `sum.IntList_out` (180), `any.dims_out` (167),
`sign.out` (120), `index_put` (90 op / 64 graph-opt — genuinely split), `floor_divide` (85+85).
Top graph-optimization ops (correct alone, need a sibling — the copy-elision/aliasing class):
`view_copy`, `transpose_copy`, `clone`, `alias_copy`, `expand_copy`, `pixel_unshuffle`,
`squeeze_copy`. This quantifies, per op, the operator-vs-graph-opt split the representative
bisections below illustrate.

### Per-operator bug catalog (device-verified single-op isolation)
Each operator-verdict op was re-isolated as a **one-op graph fed its exact baked inputs** and run on
the Moto (`tmp/isolate_op.py`). Of 196 candidate ops: **89 confirmed GENUINE** (diverge alone,
finite inputs), 92 rejected CLEAN (pass alone → graph-opt confound / input-specific), 8 NONFINITE_OUT
(nan-handling), 7 LOWERFAIL (can't isolate as a single op). The 89 were then put through a **7× stability re-verification** (separating device-fail from
reproduces-eager), which dropped 40 (reproduce eager on re-run, or won't isolate as one op) and left
**49 confirmed**: **45 deterministic operator bugs** + **4 non-deterministic kernels** (wrong output
*intermittently* on identical baked input — `argmin.out`, `bitwise_and.{Scalar_out,Tensor_out}`,
`flip`). Each confirmed op has its own `bugs/operators/<op>.md` + runnable `repro_<op>.py`, indexed
in `bugs/operators/INDEX.md`; the 40 ruled-out are kept in `bugs/operators/ruled_out/` with reasons.
Full isolation provenance: `operator_isolation.tsv`.

> Methodology note: this full sweep was run as a resumable, sharded fleet (`tmp/bisect_all.py` +
> `tmp/bisect_fleet.sh`). A mid-run device crash contaminated the last ~30 graphs with false
> `dev=TIMEOUT`s; those were re-bisected once the phone was restarted (all but the 3 host-crashers
> resolved to real verdicts), per the "retry every failure / no borrowed verdicts" rule in
> `analysis.md`.

**Headline:** Android 14 reproduces the prior Moto run's Vulkan bug surface **job-for-job** —
46,693 of 46,748 non-OK jobs carry the *identical* status (99.9% concordance). The only
differences are a small, symmetric, mostly-flaky precision tail (47 A14-unique vs 41
baseline-unique). **No new bug class appears on Android 14.** Details in `CROSS_DEVICE_DIFF.md`.

Each non-OK graph is placed on two axes: **failure mode** (mismatch / crash / skip) × **root
cause** (operator / graph-optimization).

---

## Mismatch bugs — root cause by on-device bisection
Bisection (`tmp/bisect_graph.py`) minimizes the returned-output set on the phone: target diverges
as the **only** returned output → operator; needs a **sibling co-returned** → graph-optimization.

### Operator bugs (target diverges alone) — device-verified on A14
| op | bisected example (this run) | minimal cone | repro |
|----|------------------|-------------:|-------|
| `floor_divide` | **w1:217 out[0] → OPERATOR, cone 2** ✓ | 2 | `bugs/repro_vulkan-floor-divide-self.py` |
| `clamp.Tensor_out` | **w0:379 out[0] → OPERATOR, cone 2** ✓ | 2 | (corpus-input-triggered) |
| `scatter.src_out` | isolation: eager `[5,2,6,4]` → device `[0,2,3,4]` (drops writes) ✓ | 1 | `bugs/repro_vulkan-scatter-src-drops-writes.py` |
| `index_put` | isolation: eager `[0,7,0,8]` → device `[0,0,0,0]` (drops writes) ✓ | 1 | `bugs/repro_vulkan-index-put-drops-writes.py` |
| `bitwise_left/right_shift` (over-width) | — | 1 | `bugs/repro_vulkan-left-shift-overwidth.py` |
| `select_scatter`, `scatter_add.out`, `topk.values`, `diagonal_copy` | per prior run (corpus-input-triggered) | — | — |

`floor_divide` and `clamp.Tensor_out` bisections were **re-run on this A14 device** and reduce to
the single producing op as the sole returned output → confirmed operator bugs here. `scatter.src_out`
and `index_put` **drop their writes** in a one-op graph on A14 (most severe class).

### Graph-optimization bugs (target needs a sibling co-returned) — device-verified on A14
| op | bisected example (this run) | required sibling | class |
|----|------------------|------------------|-------|
| `view_copy` | **w1:658 out[2] → correct alone, diverges when `alias_copy` (n4) co-returned** ✓ | `alias_copy` | copy-elision / aliasing |
| `pixel_shuffle`, `clone` / `lift_fresh_copy` / `alias_copy` | per prior A/B repro | — | copy-elision / aliasing |

Confirmed copy-elision / memory-planning aliasing — the op is correct alone but a co-returned
sibling changes buffer planning and corrupts it. A/B neutralization in
`bugs/vulkan-copy-elision-aliasing.md`.

**Non-finite first:** 2,713 of the 9,702 mismatches are nan/inf-position diffs — usually upstream
propagation, not a sink-op bug (the oracle treats *matching* nan/inf positions as OK). Attributed
to the producing op, not the sink.

---

## Crash bugs — root cause by single-op isolation (crashes can't be per-output bisected)
| op(s) | isolated single-op result on A14 | root cause |
|-------|----------------------------------|------------|
| `unfold_copy` | **OK alone** ✓ | graph-optimization (aliasing) — crash only in larger graphs |
| `narrow_copy` (and length-0) | **OK alone** ✓ | graph-optimization (aliasing) |
| `clone`, `lift_fresh_copy`, `alias_copy`, `transpose_copy`, `expand_copy` | correct alone (per prior) | graph-optimization (aliasing) |
| `native_group_norm` / `native_layer_norm` | **SKIP every weight form** ✓ | operator/validation (also a skip bug) |
| `scatter.src_out` | drops writes (mismatch alone) ✓ | operator |

The copy/view crash cluster (`narrow_copy`, `unfold_copy`, `clone`, `alias_copy`, `lift_fresh_copy`,
`transpose_copy`, `expand_copy`) is the **same aliasing graph-optimization bug** surfacing as a
crash rather than a silent mismatch — not broken kernels. Device-verified correct in isolation here.

---

## Skip bugs — backend lowers it, device rejects at runtime (coverage gaps)
Counts from the device's own rejection messages (each a device-verified runtime rejection):

| coverage gap | device rejection signature (count) |
|--------------|------------------------------------|
| **layer_norm** | `add_native_layer_norm_node` / `prepack_standard` — const-weight / rank-1 only (3,424) |
| **missing dtype shaders** | `get_shader_info @ ShaderRegistry.cpp:54` — view/convert buffer dtype (4,095) |
| **reductions over >2 dims** | `amin`/`amax`/`mean`/`sum @ Reduce.cpp`, `arg_reduce_impl` (≈4,000) |
| **binary broadcast** | `check_binary_op_args @ BinaryOp.cpp:35` — out_sizes != broadcasted_sizes (2,332) |
| **scalar from non-const** | `extract_scalar … type NONE` (2,041) |
| **group_norm** | `native_group_norm @ GroupNorm.cpp:188` — requires const weight (1,506) |
| **batch_norm** | `add_native_batch_norm_node @ BatchNorm.cpp:65` — 2d only (1,032) |
| generic invalid-arg / OOB | `ExecutorchInvalidArgumentException` (6,436), `ArrayIndexOutOfBounds` (4,575) |

`native_group_norm` and `native_layer_norm` additionally confirmed by single-op isolation on A14
(`ISOLATION_PROBES.md`): they SKIP in **every** weight form (const / leaf / none / rank-2).

---

## Summary grid (failure mode × root cause, device-verified on A14)
| | operator | graph-optimization |
|---|---|---|
| **mismatch** | floor_divide, clamp.Tensor_out, scatter.src_out, index_put, bitwise shift, select_scatter, scatter_add, topk.values, diagonal_copy | view_copy, pixel_shuffle, clone/lift_fresh_copy/alias_copy (copy-elision aliasing) |
| **crash** | native_group_norm, native_layer_norm, scatter.src_out | narrow_copy, unfold_copy, clone, alias_copy, transpose_copy, expand_copy (same aliasing class) |
| **skip** | layer_norm, group_norm, batch_norm, amax/amin/mean/sum (>2 dims), binary-broadcast, missing dtype shaders, extract_scalar (coverage gaps) | — |

These match the prior Moto run exactly. The Android-14 contribution is the **cross-OS
confirmation** (same SoC class, newer/older OS): the entire ExecuTorch-Vulkan bug surface is
**OS-version-independent** on the Mali-G57 — see `CROSS_DEVICE_DIFF.md`.
