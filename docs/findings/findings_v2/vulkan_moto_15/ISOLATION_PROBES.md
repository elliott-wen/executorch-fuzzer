# On-device isolation probes — Moto G54 5G

Each top culprit rebuilt as a **single-op job** (`tmp/build_culprit_probes.py`) and run on the
Moto G54. Ground truth the statistics can't give — especially for CRASH, which can't be
per-output attributed. Results are **identical to the Pixel 9**.

## Results

| probe | Moto result | verdict |
|-------|-------------|---------|
| `native_group_norm` (const / leaf / none weight) | **SKIP** — `CppException from native_group_norm` (all 3) | **GENUINE** |
| `native_layer_norm` const weight | **SKIP** — `prepack_standard` | **GENUINE** |
| `native_layer_norm` None weight | **SKIP** — `add_native_layer_norm_node` | **GENUINE** |
| `native_layer_norm` rank-2 normalized_shape | **SKIP** — `add_native_layer_norm_node` | **GENUINE** |
| `scatter.src_out` | **MISMATCH** — eager `[5,0,6,0]` vs device `[0,0,0,0]` | **GENUINE, severe** — writes dropped (see bugs/) |
| `index_put` | **MISMATCH** — eager `[0,7,0,8]` vs device `[0,0,0,0]` | **GENUINE, severe** — writes dropped (see bugs/) |
| `clamp.out` (scalar) — control | OK | — |
| `clamp.Tensor_out` (equal-shape bounds) | **OK** | **CONDITIONAL** — 56% in corpus; trigger = broadcast/dynamic bounds |
| `select_scatter` (float) | **OK** | **CONDITIONAL** — 51% in corpus is **all dtype**; float passes |
| `bitwise_left_shift` small/large (int64) | **OK** | **CONDITIONAL** — needs bool / int32 overflow / specific amounts |
| `unfold_copy` | **OK** | **CONFOUND** — 40% crash in corpus, correct alone |
| `narrow_copy` (and empty length-0) | **OK** | **CONFOUND** — 36% crash in corpus, correct alone |

## Interpretation (same as Pixel 9)
- **Genuine kernel/validation bugs:** `native_group_norm`, `native_layer_norm`, `scatter.src_out`
  (writes dropped, element 0 →0), `index_put` (writes dropped, element 0 →0).
- **Confounds:** `unfold_copy`, `narrow_copy` are correct alone — their high in-corpus crash rate
  is the **memory-planning / aliasing bug** (see `bugs/vulkan-copy-elision-aliasing.md`), not a
  broken kernel.
- **Conditional:** `clamp.Tensor_out` (broadcast/dynamic bounds), `select_scatter` (integer
  dtype), `bitwise_left_shift` (bool/overflow) — real in corpus, minimal probe doesn't hit the
  trigger.

That the verdicts match the Pixel 9 op-for-op confirms the bugs are **delegate-level**.
