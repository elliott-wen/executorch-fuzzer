# Portable corpus — SKIP reasons (graceful kernel refusals)

Census of all **37,746 SKIP verdicts (32.2% of the run)**. A SKIP is the *safe* outcome:
the portable kernel ran a runtime guard (`ET_KERNEL_CHECK` / dtype dispatch) and returned a
non-OK `Error` to the client **without crashing**. These are not correctness bugs — they are
the portable backend honestly reporting "I don't run this op for these arguments." Their value
is as a **coverage map** (which ops/args a portable-only deployment cannot execute) and as the
contrast to the [CRASH](../crash/) cases — the same invalid-arg situations that reach an
*unguarded* op abort instead.

Unlike crashes/mismatches, skips carry the full error in the TSV `reason` column, so no
localizer is needed — these are grouped directly by `(operator, firing guard)`.
Aggregator: `tmp/run_portable/skip_aggregate.py` (raw output in `_aggregate.txt`).

## Distribution by category

| count | share | category | meaning | file |
|---:|---:|---|---|---|
| 11,141 | 29.5% | unimplemented arg path | op exists but a flag/mode isn't wired up | [02-unimplemented-arg-paths.md](02-unimplemented-arg-paths.md) |
| 10,992 | 29.1% | shape / arg validation | kernel correctly rejects malformed input (wrong rank/dim/size) | [03-input-validation-guards.md](03-input-validation-guards.md) |
| 10,483 | 27.8% | dtype coverage gap | kernel implements only a subset of dtypes | [01-dtype-coverage-gaps.md](01-dtype-coverage-gaps.md) |
| 2,990 | 7.9% | other guard | hard-disabled path (`Check(false)`) + out-resize failures | [05-other-guards.md](05-other-guards.md) |
| 2,140 | 5.7% | type-promotion guard | stricter promotion contract than eager | [04-type-promotion-guards.md](04-type-promotion-guards.md) |

Source-level spotlight on the dtype-coverage category: [06-complex-dtype-support.md](06-complex-dtype-support.md)
— portable is **real-only in ~162 of 172 kernels**; only ~10 ops support complex, which is the root of
the dominant `_conj_physical` SKIP.

## Top operators

| count | operator | dominant reason |
|---:|---|---|
| 10,023 | `_conj_physical` | Unhandled dtype (complex-only kernel; fuzzer feeds real dtypes) |
| 4,821 | `expand_copy` | `implicit==true` not implemented |
| 4,775 | `copy` | `non_blocking==true` not supported |
| 4,461 | `native_group_norm` | `in.size(0)==N` / `in.size(1)==C` shape checks |
| 3,069 | `_pdist_forward` | input must be rank 2 |
| 2,878 | `_log_softmax` | `Check(false)` — hard-disabled path |
| 1,695 | `cumsum` | `dim` out of range |
| 1,551 | `_fft_r2c` | `onesided=False` / contiguity not supported |
| 933 | `add.Scalar` | common_type / alpha promotion guard |
| 785 | `_adaptive_avg_pool2d` | `output_size > 0` |
| 619 | `_native_batch_norm` | input/running-stat dtype must match |
| 602 | `arange` | non-extractable scalar bound |
| 593 | `convolution` | input rank check |
| 430 | `scatter`/`scatter_add` | `index` must be Long |

## How to read this vs CRASH

The actionable signal is the **inverse** of the crash census: the ops that *crash*
([narrow_copy](../crash/01-narrow_copy-negative-dim.md), [unfold_copy](../crash/02-unfold_copy-zero-dim.md))
are missing the very kind of guard that the ops here run correctly. The "real coverage gap"
candidates a production model could hit are the **unimplemented arg paths** and the
**`_log_softmax` hard-disabled path** — the rest are mostly the kernel correctly rejecting
fuzzer-malformed inputs.
