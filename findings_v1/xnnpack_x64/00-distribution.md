# XNNPACK corpus — differential-fuzz run: distribution + comparison to portable

Run of the **XNNPACK-lowered** `.pte` corpus (`corpus/xnnpack/`, 100,032 graphs) through
`xnnpack_client` — the same in-process ExecuTorch host runtime, but each program lowered to the
**XNNPACK delegate** (Google's optimized fp32/quantized CPU kernels). Ops XNNPACK delegates run
its kernels; everything else falls back to the **portable** kernels. Reference = eager PyTorch.

## Top-level verdicts (vs portable)

| verdict | xnnpack | share | portable | share |
|---|---:|---:|---:|---:|
| OK | 49,521 | 49.5% | 63,480 | 54.1% |
| MISMATCH | 11,024 | 11.0% | 10,066 | 8.6% |
| CRASH | 5,157 | 5.2% | 5,996 | 5.1% |
| SKIP | 34,330 | 34.3% | 37,746 | 32.2% |
| TIMEOUT | 0 | — | 0 | — |
| **total** | **100,032** | | **117,288** | |

**XNNPACK mismatches more (11.0% vs 8.6%)** — the delegate *adds* divergences on top of the
shared portable-kernel bugs. Of programs that ran to completion (60,545), **18.2% mismatched**
(vs 13.7% for portable).

## The central result: shared vs XNNPACK-specific

A per-op mismatch-propensity comparison (rate = how often an op is the flagged mismatch output,
xnnpack vs portable) splits the mismatches cleanly:

### Shared with portable (same rate + kind → not delegated, falls back to portable kernels)

| op | xnn rate | port rate | → see portable finding |
|---|---:|---:|---|
| `select_scatter` (dtype) | 22.0% | 22.5% | [bugs/select_scatter-dtype.md](../portable/bugs/select_scatter-dtype.md) |
| `bitwise_left_shift` (delta) | 14.8% | 15.2% | [bugs/bitwise-shift-negative.md](../portable/bugs/bitwise-shift-negative.md) |
| `remainder` | 3.9% | 4.0% | [bugs/remainder-sign.md](../portable/bugs/remainder-sign.md) |
| `floor_divide` (nonfinite) | 6.0% | 6.3% | [bugs/floor_divide.md](../portable/bugs/floor_divide.md) |
| `sum` | 7.4% | 7.5% | [bugs/sum-cast-order.md](../portable/bugs/sum-cast-order.md) |
| `sign` | 6.5% | 6.4% | [bugs/sign-nan.md](../portable/bugs/sign-nan.md) |
| `prod` | 3.0% | 3.3% | [bugs/prod-pow-fmod-rshift.md](../portable/bugs/prod-pow-fmod-rshift.md) |
| `_native_batch_norm` | 22.0% | 22.5% | [bugs/normalization-bornhere.md](../portable/bugs/normalization-bornhere.md) |
| `max_pool2d_with_indices_backward` | 16.8% | 17.5% | [bugs/maxpool2d-backward.md](../portable/bugs/maxpool2d-backward.md) |

→ These need no re-investigation; the portable root causes apply unchanged.

### NEW — XNNPACK-delegate-specific (rate spikes, dominated by **non-finite**)

| op | xnn rate | port rate | ×up | dominant kind |
|---|---:|---:|---:|---|
| `sqrt` | 5.7% | 0.4% | **14×** | nonfinite 298 |
| `exp` | 6.3% | 0.5% | **12×** | nonfinite 321 |
| `rsqrt` | 4.7% | 0.8% | 6× | nonfinite 249 |
| `relu` | 3.7% | 0.3% | 12× | nonfinite 171 |
| `clamp` | 2.4% | 0.4% | 6× | nonfinite 216 |
| `hardtanh` | 4.0% | 0.4% | 10× | nonfinite 177 |
| `minimum` | 3.9% | 0.7% | 6× | nonfinite 206 |
| `maximum` | 3.1% | 0.6% | 5× | nonfinite 150 |

These are all **XNNPACK-delegated fp32 activation/elementwise ops**, and they diverge on
**non-finite handling**. Localizer-confirmed born-here (inputs agree, output differs):
- `w0:49` — `hardtanh(nan)` → xnnpack **-2** vs eager **nan**
- `w0:734` — `minimum(nan, 1)` → xnnpack **1** vs eager **nan**

→ Root cause (the clamp/min/max family): XNNPACK's SIMD `min`/`max`/clamp instructions return
the **non-NaN operand** (IEEE `fmin`/`fmax` / `max(x,0)` for relu), so they **launder NaN to a
finite value**, whereas eager PyTorch **propagates NaN**. The `sqrt`/`exp`/`rsqrt` spikes are
the same family of fast-approximation kernels handling nan/inf/overflow differently. Full
investigation: [bugs/xnnpack-activation-nonfinite.md](bugs/xnnpack-activation-nonfinite.md).

## SKIP: also mostly shared, plus one new XNNPACK class

The skip census mirrors portable (same operators: `_conj_physical`, `expand_copy`, `copy`,
`native_group_norm`, …) — these are portable-kernel guards that fire regardless of delegation.
**New in xnnpack:** **3,487 `Failed to load method forward`** — graphs whose XNNPACK-lowered
`.pte` cannot even be loaded by the runtime (a delegate serialization/partition failure, not a
runtime refusal). See [bugs/xnnpack-load-failure.md](bugs/xnnpack-load-failure.md).

Raw data: `tmp/run_xnnpack/` (`skip_reasons_xnnpack.tsv`, `localized.jsonl`, `crashes.jsonl`,
aggregators). Crash analysis: [crash/](crash/). Skip analysis: [skip/](skip/).
