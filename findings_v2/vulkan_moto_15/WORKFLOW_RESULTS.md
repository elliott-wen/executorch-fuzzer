# Workflow results — Moto G54 5G (analysis.md applied to the existing skip-log)

Ran the `analysis.md` workflow on the Moto using the already-collected `tmp/vulkan_moto_skip.tsv`
(feed step skipped). Every classification below is **device-verified**: mismatch root causes by
graph bisection on the phone (`tmp/bisect_graph.py`), crash/skip by single-op isolation.

## Triage (from the skip-log)
| outcome | count |
|---|---:|
| OK | 30,059 |
| MISMATCH | 9,703 |
| CRASH | 3,182 |
| SKIP | 33,857 |
| TIMEOUT | 0 |

Each non-OK graph is placed on two axes: **failure mode** (mismatch / crash / skip) × **root
cause** (operator / graph-optimization).

---

## Mismatch bugs — root cause by device bisection

Bisection minimizes the returned-output set on the phone; if the target output diverges as the
**only** returned output → operator, if it needs a **sibling co-returned** → graph-optimization.

### Operator bugs (target diverges alone)
| op | bisected example | minimal cone | repro |
|----|------------------|-------------:|-------|
| `floor_divide` | w1:217 out[0] | 2 | `bugs/repro_vulkan-floor-divide-self.py` |
| `bitwise_left_shift` / `bitwise_right_shift` (over-width) | — | 1 | `bugs/repro_vulkan-left-shift-overwidth.py` |
| `clamp.Tensor_out` | w0:379 out[0] | 2 | (corpus-input-triggered; minimal cone isolated) |
| `select_scatter` | w100:617 out[1] | 5 | (corpus-input-triggered) |
| `scatter.src_out` | (isolation) | 1 | `bugs/repro_vulkan-scatter-src-drops-writes.py` |
| `index_put` | (isolation) | 1 | `bugs/repro_vulkan-index-put-drops-writes.py` |
| `scatter_add.out` | w119:130 out[2] | 9 | (corpus-input-triggered) |
| `topk.values` | w0:181 out[0] | 3 | (corpus-input-triggered) |
| `diagonal_copy` | w101:113 out[1] | 3 | (corpus-input-triggered) |

`clamp.Tensor_out` and `select_scatter` were "couldn't isolate" in the first round (minimal
hand-built cases passed). Bisection of real failing graphs shows they **do** diverge as the sole
output on corpus inputs → confirmed operator bugs, just input-condition-specific.

### Graph-optimization bugs (target needs a sibling co-returned)
| op | bisected example | required sibling | class |
|----|------------------|------------------|-------|
| `view_copy` | w1:658 out[2] | `alias_copy` | copy-elision / aliasing |
| `pixel_shuffle` | w1:248 out[0] | sibling node | copy-elision / aliasing |
| `clone` / `lift_fresh_copy` / `alias_copy` | (A/B repro) | — | copy-elision / aliasing |

These are the **copy-elision / memory-planning aliasing** class — the op is correct alone but a
sibling changes buffer planning and corrupts it. Confirmed by the A/B repro
(`bugs/vulkan-copy-elision-aliasing.md`): swapping the copy op for a fresh-buffer op removes the
divergence.

---

## Crash bugs — root cause by isolation (crashes can't be per-output bisected)

| op(s) | isolated single-op result | root cause |
|-------|---------------------------|------------|
| `unfold_copy`, `narrow_copy` | **correct alone** | graph-optimization (aliasing) — crash only in larger graphs |
| `clone`, `lift_fresh_copy`, `alias_copy`, `transpose_copy`, `expand_copy` | correct alone | graph-optimization (aliasing) |
| `native_group_norm` | SKIP (rejected) all weight forms | operator/validation (also a skip bug) |
| `native_layer_norm` | SKIP (rejected) | operator/validation (also a skip bug) |
| `scatter.src_out` | drops writes (mismatch alone) | operator |

The copy/view crash cluster (`unfold_copy` 40%, `narrow_copy` 36%, plus `clone`/`alias_copy`/
`lift_fresh_copy`) is the **same aliasing graph-optimization bug** surfacing as a crash rather than
a silent mismatch — not broken kernels.

---

## Skip bugs — backend lowers it, device rejects at runtime (coverage gaps)

Counts from the device's own rejection messages in the skip-log (each is a device-verified
runtime rejection). Top categories:

| coverage gap | device rejection (count) |
|--------------|--------------------------|
| **layer_norm** | `add_native_layer_norm_node` — requires const weight / rank-1 (3,424) |
| **missing dtype shaders** | `get_shader_info … view_convert_buffer_{float,half}_uint*` (≈3,500) |
| **reductions over >2 dims** | `amin`/`amax`/`arg_reduce_impl` (≈3,750) |
| **binary broadcast** | `check_binary_op_args` — out_sizes != broadcasted_sizes (2,332) |
| **scalar from non-const** | `extract_scalar … type NONE` (2,041) |
| **group_norm** | `native_group_norm` — requires const weight (1,506) |
| **batch_norm** | `add_native_batch_norm_node` — 2d only (1,032) |

`native_group_norm` and `native_layer_norm` additionally confirmed by single-op isolation
(`ISOLATION_PROBES.md`): they SKIP in every weight form.

---

## Summary grid (failure mode × root cause, device-verified)

| | operator | graph-optimization |
|---|---|---|
| **mismatch** | floor_divide, bitwise shift, clamp.Tensor_out, select_scatter, scatter.src_out, index_put, scatter_add, topk.values, diagonal_copy | view_copy, pixel_shuffle, clone/lift_fresh_copy/alias_copy (copy-elision aliasing) |
| **crash** | native_group_norm, native_layer_norm, scatter.src_out | unfold_copy, narrow_copy, clone, alias_copy, transpose_copy, expand_copy (same aliasing class) |
| **skip** | layer_norm, group_norm, batch_norm, amax/amin, binary-broadcast, missing dtype shaders, extract_scalar (coverage gaps) | — |

Reproduce any verdict: `python tmp/bisect_graph.py <job_id> <out_index>` (mismatch root cause),
or run the matching `bugs/repro_*.py` (needs a connected Vulkan worker on `127.0.0.1:15554`).
