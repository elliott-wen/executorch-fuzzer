# Vulkan (Galaxy, on-device) differential-fuzz — distribution & localization

`corpus/vulkan` (100,032 auto-generated graphs) lowered with the ExecuTorch **Vulkan** delegate and
executed on a real **Samsung Galaxy** phone (Adreno GPU) — jobs dispatched from the host broker over
a `rathole` tunnel, each output diffed against eager PyTorch.

## Verdicts

| verdict | count | share |
|---|---:|---:|
| OK | 13,331 | 13.3% |
| **MISMATCH** | **8,202** | 8.2% |
| **CRASH** | **6,214** | 6.2% |
| SKIP (graceful refusal) | 72,285 | 72.3% |
| TIMEOUT | 0 | — |

Of the ~27.7k graphs that actually executed (non-SKIP), **~52% diverged** from eager. The Vulkan
delegate refuses most of the corpus up front (72k SKIPs) — see [skip/README.md](skip/README.md).

## MISMATCH — by divergence kind (feeder)

| kind | n |
|---|---:|
| non-finite (NaN/Inf positions differ) | 2,406 |
| max\|delta\| 1..1e6 (rtol=0, strict) | 1,833 |
| max\|delta\| 1..1e6 (rtol>0) | 1,743 |
| max\|delta\| <1 (small) | 786 |
| max\|delta\| ≥1e15 (overflow-ish) | 907 |
| dtype int64 vs float32/float16 | 299 |
| dtype bool vs int64 | 33 |
| shape mismatch | 86 |

The ≥1e15 bucket and the int64 dtypes are the **int64-not-representable** family (`9.223e18` =
INT64 limits). The non-finite bucket is fp16 overflow + dropped special values.

## MISMATCH — first-divergence localization (the useful view)

An **on-device first-divergence localizer** ([../tmp/run_vulkan_galaxy_26/vulkan_localize.py]) re-lowered
each mismatch graph returning **every intermediate node**, ran it on the phone, and found the first
node whose value differs from eager = the **born-here root op**. Sample: 2,974 jobs (≈30 per
flagged-op), **2,497 localized OK**.

**83% of mismatches are *inherited*** — the feeder-flagged output is a downstream symptom; the real
divergence was born several ops upstream. Collapsing to born-here roots reduces ~8k mismatches to
~20 root ops:

| born-here root op | kind | ~n | root cause |
|---|---|---:|---|
| `_to_copy` | delta | 402 | [int64 truncation](bugs/vulkan-int64-truncation.md) (INT64_MIN→0) |
| `clamp` / `hardtanh` | delta/nonfinite | 335 / 38 | **inherited** int64/fp16 — [not a clamp bug](bugs/vulkan-clamp.md) |
| `pow` | nonfinite/delta | 143 | [fp16 + special values](bugs/vulkan-fp16-overflow.md) |
| `bitwise_left_shift` / `bitwise_right_shift` | delta | 141 / 49 | [shared: negative-shift UB](bugs/shared-with-portable-xnnpack.md) |
| `index_put` | delta | 132 | [vulkan int64-index aliasing](bugs/vulkan-index-put.md) |
| `floor_divide` | delta | 102 | [vulkan trunc-toward-zero](bugs/vulkan-floor-divide.md) |
| `remainder` | delta | 93 | [shared: sign-of-divisor](bugs/shared-with-portable-xnnpack.md) |
| `sum` / `prod` / `mean` | delta/nonfinite | 74 / 45 / 27 | [fp16 accumulate + cast order](bugs/vulkan-fp16-overflow.md) |
| `select_scatter` | dtype | 69 | [shared: where-promotion decomp](bugs/shared-with-portable-xnnpack.md) |
| `sign` | nonfinite | 38 | [NaN→NaN, should be 0](bugs/vulkan-sign-nan.md) |
| `_softmax` / `_log_softmax` | nonfinite | 37 / 26 | [NaN scrub + denom clamp](bugs/vulkan-logsoftmax-nonfinite.md) |
| `relu` / `minimum` / `tanh` | nonfinite | 33 / 28 / 32 | [min/max/clamp drop NaN](bugs/vulkan-tanh-nonfinite.md) |
| `_native_batch_norm_legit` | nonfinite | 36 | fp16 normalization |
| `broadcast_to` | shape | 19 | [generator-artifact](bugs/vulkan-broadcast-shape.md) |

Localizer caveat: `inherited` is computed by op-*name* (`root_op != flagged_op`), so a graph with two
nodes of the same op can self-alias to `inherited:false` — a measurement artifact, not a born-here
claim. The localizer also cannot attribute **memory-planning** bugs to a node (see
[WHY-CORPUS-MISSED-IT.md](WHY-CORPUS-MISSED-IT.md) and the
[copy-elision aliasing bug](bugs/vulkan-copy-elision-aliasing.md)).

## CRASH — op enrichment (no device signature available)

A native abort kills the phone executor outright; the broker only **infers** CRASH from the dropped
binding ([ExecutorService.kt:68-69](../../android/app/src/main/java/com/fuzzer/etclient/ExecutorService.kt#L68-L69)).
No abort message is captured over the wire and there is no adb path to the remote phone — so crashes
are characterized by **op-presence enrichment vs the full-corpus baseline**, not per-crash signatures:

| op | in crashes | p(crash) | p(baseline) | lift |
|---|---:|---:|---:|---:|
| `unfold_copy` | 1,880 | 0.303 | 0.103 | **2.93×** |
| `replication_pad3d` | 289 | 0.047 | 0.018 | 2.58× |
| `split_with_sizes_copy` | 676 | 0.109 | 0.051 | 2.14× |
| `narrow_copy` | 2,463 | 0.396 | 0.234 | 1.70× |
| `pixel_unshuffle` / `tril` | 591 / 403 | — | — | 1.70× / 1.69× |
| `clone` / `alias_copy` / `lift_fresh_copy` | ~1,080 each | — | — | ~1.73× |

Most enriched ops are **not delegated to Vulkan** (they run on portable CPU and act as boundary
markers); the population is dominated by degenerate/unsupported shapes (83% int64-out, 68% >4D, 57%
zero-numel). Full analysis: [crash/README.md](crash/README.md). The `clone`/`alias_copy`/
`lift_fresh_copy` enrichment ties to the [copy-elision aliasing bug](bugs/vulkan-copy-elision-aliasing.md).

## Artifacts (`tmp/run_vulkan_galaxy_26/`)

- `skip_reasons_mobile.tsv` — raw feeder output (status, job, reason, op-chain), 86,701 failing rows
- `analyze.py`, `skip_aggregate.py` → `skip_census.txt`, `per_op_propensity.py` → `propensity.txt`
- `vulkan_localize.py` (on-device first-divergence) → `localized.jsonl`, `aggregate_localize.py` → `localize_aggregate.txt`
- `check_copy_alias.py`, `why_missed.py` — the aliasing-bug confirmation + miss analysis
