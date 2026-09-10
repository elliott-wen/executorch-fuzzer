# XNNPACK-corpus differential-fuzz — findings

Run of the **XNNPACK-lowered** `.pte` corpus (`corpus/xnnpack/`, 100,032 graphs) through
`xnnpack_client`, diffed against eager PyTorch. Same harness/methodology as the
[portable run](../portable/README.md) — distribution analysis, a node-level **first-divergence
localizer** (lowering each graph through the XNNPACK delegate), and a **crash localizer**.

## Verdicts (vs portable)

| verdict | xnnpack | share | portable share |
|---|---:|---:|---:|
| OK | 49,521 | 49.5% | 54.1% |
| MISMATCH | 11,024 | 11.0% | 8.6% |
| CRASH | 5,157 | 5.2% | 5.1% |
| SKIP | 34,330 | 34.3% | 32.2% |

Details + the full portable comparison: [00-distribution.md](00-distribution.md).

## The one-line story

**Most findings are shared with portable** (XNNPACK only delegates a subset of ops; everything
else falls back to the *same* portable kernels and reproduces the *same* bugs). The XNNPACK
delegate adds **two new bug families** of its own, which is why mismatches rise 8.6% → 11.0%.

### Shared with portable (no re-investigation needed)
- **All crashes** ([crash/](crash/README.md)) — `narrow_copy` neg-dim, `unfold_copy` 0-D,
  integer divide-by-zero SIGFPE — identical profile, attributed to the portable kernels.
- **Most mismatches** — localizer born-here roots match portable: `select_scatter` (dtype),
  `bitwise_left_shift`/`right_shift`, `remainder`, `sum`, `prod`, `sign`, `batch_norm`,
  `cdist`/`pdist`, `group_norm`/`layer_norm`/`log_softmax`, `floor_divide`, the `broadcast_to`
  shape-aliasing artifact. → see [portable/bugs/](../portable/bugs/README.md).
- **Most skips** ([skip/](skip/README.md)) — same portable-kernel guards (`_conj_physical`
  coverage gap, `expand implicit`, `copy non_blocking`, validation guards, …).

### NEW — XNNPACK-delegate-specific bugs → [bugs/](bugs/)

Each bug is split into its own file with a co-located **buggy model graph** `graph_<bug>.py` (see
[bugs/README.md](bugs/README.md)); the activation cluster is now three per-mechanism files.

| bug | scope | root cause | file |
|---|---:|---|---|
| **(A) clamp/min/max NaN-launder** | ~thousands born-here | `relu`/`clamp`/`hardtanh`/`minimum`/`maximum` lower to SIMD `fmin`/`fmax` that returns the **non-NaN operand** → NaN laundered to a finite bound. Eager propagates NaN. | [bugs/xnnpack-clamp-minmax-nan.md](bugs/xnnpack-clamp-minmax-nan.md) |
| **(B) sqrt/rsqrt domain** | 5.7%/4.7% op propensity | `sqrt(x)=x·rsqrt(x)` via fast rsqrt → `sqrt(-x)=-0` (not `NaN`); `rsqrt(±0)≠±inf`. | [bugs/xnnpack-sqrt-rsqrt-domain.md](bugs/xnnpack-sqrt-rsqrt-domain.md) |
| **(C) exp NaN/overflow** | 6.3% op propensity | rational-poly `exp` has no NaN/overflow guard → `exp(nan)=0`. | [bugs/xnnpack-exp-nonfinite.md](bugs/xnnpack-exp-nonfinite.md) |
| **`.pte` fails to load** | 3,487 (the new SKIP class) | XNNPACK **partition over-inclusion**: delegates ops with tensors exceeding `XNN_MAX_TENSOR_DIMS=6` (rank-7 from `pixel_shuffle`/`unshuffle` decomp) → `xnn_define_tensor`/`static_transpose` reject the subgraph at load → whole program unloadable. | [bugs/xnnpack-load-failure.md](bugs/xnnpack-load-failure.md) |

## Why these are XNNPACK-only (and the method that proved it)

A per-op **mismatch-propensity comparison** (rate that an op is the flagged mismatch output,
xnnpack vs portable) cleanly separated shared from new: shared ops have identical rates
(e.g. `select_scatter` 22.0% vs 22.5%), while the activation ops spike only in xnnpack
(`sqrt` 5.7% vs 0.4%, `exp` 6.3% vs 0.5%). The first-divergence localizer — which lowers each
graph through the **XNNPACK** delegate and finds the first node that diverges from eager —
then confirmed those ops as **born-here** (e.g. `hardtanh(nan)=−2` vs eager `nan`), proving the
divergence originates in the XNNPACK kernel, not upstream.

## Highest-value XNNPACK-side fixes
1. **NaN propagation in XNNPACK clamp/minmax/relu** (or don't partition these ops when strict
   IEEE non-finite behavior is required) — the largest new cluster.
2. **Rank guard in the XNNPACK partitioner** (`dim() > XNN_MAX_TENSOR_DIMS` → keep portable) —
   eliminates all 3,487 load failures; the constant already exists.
3. Everything else: apply the [portable bug fixes](../portable/bugs/README.md) — they cover the
   shared crash and mismatch sets unchanged.

Raw data + aggregators: `tmp/run_xnnpack/` (`skip_reasons_xnnpack.tsv`, `localized.jsonl`,
`crashes.jsonl`, `_mm_aggregate.txt`, `crash/_aggregate.txt`, `skip/_aggregate.txt`).
