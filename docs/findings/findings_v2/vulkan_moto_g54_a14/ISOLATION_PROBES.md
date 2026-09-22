# On-device isolation probes — Moto G54 5G, Android 14

Each top culprit rebuilt as a **single-op job** (`tmp/build_culprit_probes.py`, corpus
`tmp/culprit_probe`) and run on this A14 phone. Ground truth the statistics can't give —
especially for CRASH, which can't be per-output attributed. **All results match the prior Moto /
Pixel 9 runs.**

| probe | A14 result | verdict |
|-------|------------|---------|
| `native_group_norm` (const / leaf / none weight) | **SKIP** — `CppException from native_group_norm` (all 3) | **GENUINE** coverage gap |
| `native_layer_norm` const weight | **SKIP** — `prepack_standard` | **GENUINE** |
| `native_layer_norm` None weight | **SKIP** — `add_native_layer_norm_node` | **GENUINE** |
| `native_layer_norm` rank-2 normalized_shape | **SKIP** — `add_native_layer_norm_node` | **GENUINE** |
| `scatter.src_out` | **MISMATCH** — eager `[5,0,6,0]` vs device `[0,0,0,0]` | **GENUINE, severe** — writes dropped (`bugs/`) |
| `index_put` | **MISMATCH** — eager `[0,7,0,8]` vs device `[0,0,0,0]` | **GENUINE, severe** — writes dropped (`bugs/`) |
| `clamp.out` (scalar) — control | **OK** | — |
| `clamp.Tensor_out` (equal-shape bounds) | **OK** | **CONDITIONAL** — diverges on corpus inputs (bisect w0:379) |
| `select_scatter` (float) | **OK** | **CONDITIONAL** — corpus failures are dtype-specific |
| `bitwise_left_shift` small/large (int64) | **OK** (int64 not partitioned → portable fallback) | **CONDITIONAL** — needs bool/int32 over-width |
| `unfold_copy` | **OK** | **CONFOUND** — crash co-occurrence, correct alone |
| `narrow_copy` (and length-0) | **OK** | **CONFOUND** — crash co-occurrence, correct alone |

## Interpretation (same as prior Moto / Pixel 9)
- **Genuine kernel/validation bugs:** `native_group_norm`, `native_layer_norm` (skip), `scatter.src_out`,
  `index_put` (drop writes). Reproduce in a single-op graph.
- **Conditional operator bugs:** `clamp.Tensor_out`, `select_scatter`, bitwise shift — pass on simple
  inputs, diverge on specific corpus dtypes/shapes; isolated via per-job bisection, not flat probes.
- **Confounds (NOT filed as operator crashes):** `unfold_copy`, `narrow_copy`, and the copy/alias
  cluster are correct alone — their crashes are the graph-optimization aliasing bug in disguise.
