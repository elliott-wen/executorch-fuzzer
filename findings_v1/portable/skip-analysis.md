# SKIP analysis — portable kernel coverage gaps (37,746 = 32.2%)

A **SKIP** is a *graceful* refusal: the portable kernel ran a runtime guard
(`ET_KERNEL_CHECK` / dtype switch) and returned a non-OK `Error` to the client
**without crashing**. These are not correctness bugs — they are the portable backend
honestly reporting "I don't implement this op for these arguments." They are valuable
as a **coverage map**: the ops/arg-shapes a portable-only deployment cannot run.

Source: `tmp/run_portable/skip_reasons_portable.tsv` (rows with status `SKIP`),
clustered by `(operator, guard that fired)`.

## Distribution (top guards)

| count | operator | guard / reason | category |
|---:|---|---|---|
| 10,023 | `aten::_conj_physical.out` | Unhandled dtype | **dtype coverage** — portable `_conj_physical` only handles complex; fuzzer feeds real dtypes |
| 4,821 | `aten::expand_copy.out` | `implicit == true` unimplemented | **arg coverage** — implicit-expand path not implemented |
| 4,754 | `aten::copy.out` | `non_blocking == false` check | **arg coverage** — non_blocking flag guard |
| 3,839 | `aten::native_group_norm.out` | `in.size(0)` check | **shape guard** — N/group divisibility / size constraints |
| 2,872 | `aten::_log_softmax.out` | `Check failed: false` | **unimplemented** — half/unsupported dtype path hard-disabled |
| 2,865 | `aten::_pdist_forward.out` | input must be rank 2 | **shape guard** — pdist requires 2-D |
| 1,693 | `aten::cumsum.out` | `dim` out of range | **arg guard** — invalid reduction dim |
| 1,170 | `aten::_fft_r2c.out` | `onesided` constraint | **arg coverage** — r2c onesided path |
| 930 | `aten::add.Scalar_out` | common_type/alpha type check | **type-promotion guard** |
| 785 | `aten::_adaptive_avg_pool2d.out` | `output_size > 0` | **arg guard** |
| 618 | `aten::_native_batch_norm_legit_no_training.out` | dtype-consistency check | **dtype guard** |
| 613 | `aten::native_group_norm.out` | `in.size(1)` (channels/group) | **shape guard** |
| 592 | `aten::convolution.out` | `in.dim()` rank check | **shape guard** |
| 554 / 38 | `aten::arange.start_out` | `extract_scalar(start/end)` | **arg guard** — non-extractable scalar |
| 375 | `aten::_fft_r2c.out` | contiguity (`all_contiguous \|\| channels_last`) | **layout guard** |
| 214 / 114 / 101 | `aten::scatter{.value,.src,_add}.out` | `index.scalar_type()` must be long | **dtype guard** |
| 201 | `aten::_pdist_forward.out` | Unhandled dtype | **dtype coverage** |
| 58 / 24 / 20 / 20 | `permute_copy`, `bitwise_or.Scalar`, `bitwise_{left,right}_shift` | Unhandled dtype | **dtype coverage** |
| 51 / 39 | `div.out`, `sub.out` | `error == Error::Ok` (internal) | propagated upstream error |
| 27 / 26 | `remainder.Scalar`, `fmod.Scalar` | integral common_type rejected | **type guard** |

(Full long-tail in the TSV; the 30 rows above cover the overwhelming majority.)

## Interpretation

The SKIP surface falls into four buckets:

1. **dtype coverage gaps** (largest — `_conj_physical` alone is 10k, 27% of all SKIPs).
   The portable kernel implements a subset of dtypes (e.g. `_conj_physical` is meaningful
   only for complex; the fuzzer feeds float/long and the kernel refuses). Mostly *expected*
   — a real model would not call `_conj_physical` on a real tensor — but it documents which
   dtype paths are absent.
2. **unimplemented arg paths** — `expand_copy implicit==true`, `copy non_blocking`,
   `_fft_r2c onesided`. The op exists but a flag/mode is not wired up.
3. **shape / arg guards** — `group_norm`, `convolution`, `pdist`, `adaptive_avg_pool2d`,
   `cumsum dim`. These are the kernel correctly rejecting malformed inputs the fuzzer
   produced (negative/zero sizes, wrong rank, out-of-range dim). **Working as intended.**
4. **type-promotion guards** — `add.Scalar`, `remainder`/`fmod` integral rejection,
   `scatter` index-must-be-long. The kernel enforces a stricter promotion contract than
   eager; worth noting these refuse rather than silently mis-promote.

## Why SKIP ≠ bug, but worth tracking

SKIPs are the **safe** outcome — contrast with the CRASH clusters
([crash-narrow_copy](crash-narrow_copy.md), [crash-unfold_copy](crash-unfold_copy.md)),
which are the *same kind* of invalid-arg situation but reach an op that **aborts** instead
of running a guard. The actionable signal here is the inverse: the ops that crash are
missing the bounds/dtype guards that these SKIPping ops have. The two biggest "real
coverage gap" candidates (vs. fuzzer-only noise) are the **unimplemented arg paths**
(`expand_copy implicit`, `copy non_blocking`, `_fft_r2c onesided`) and `_log_softmax`'s
hard-disabled path — those could be hit by real models.
