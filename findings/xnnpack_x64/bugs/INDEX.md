# Confirmed XNNPACK-delegate bugs — index

Device-verified on the injected 87,823-graph run. All ran on the XNNPACK delegate (`ops≥1`).

## MISMATCH
| operator | count | eager → device | file |
|---|---:|---|---|
| `sqrt` | 353 | `sqrt(neg)`→-0.0 · `sqrt(inf)`→NaN | [xnnpack-sqrt-domain.md](xnnpack-sqrt-domain.md) |
| `gelu` | 92 | `gelu(inf)`: NaN → inf | [xnnpack-gelu-nonfinite.md](xnnpack-gelu-nonfinite.md) |
| `relu` / `hardtanh` | 4 | `NaN` → clamp bound (fp16 SIMD fmax/fmin launder) | [xnnpack-activation-nonfinite.md](xnnpack-activation-nonfinite.md) |

## MISMATCH — confirmed by direct trigger (corpus under-samples the value)
These are delegated to xnnpack and device-verified via a hand-fed trigger input; the uniform-random
boundary injection rarely draws the specific value each needs, so the corpus reports ~0 hits (see
[../README.md](../README.md)).
| operator | eager → device | trigger | file |
|---|---|---|---|
| `exp` | `exp(NaN)`: NaN → 0.0 | `NaN` | [xnnpack-exp-nonfinite.md](xnnpack-exp-nonfinite.md) |
| `rsqrt` | `rsqrt(0)`: inf → NaN | `±0` | [xnnpack-rsqrt-nonfinite.md](xnnpack-rsqrt-nonfinite.md) |
| `minimum` / `maximum` | `min/max(NaN,x)`: NaN → x | `NaN` in winning operand | [xnnpack-minmax-nan-launder.md](xnnpack-minmax-nan-launder.md) |

## SKIP (delegate rejects at load/execute)
| operator | count | reason | file |
|---|---:|---|---|
| `pixel_shuffle` / `pixel_unshuffle` | 99 | rank-7 > `XNN_MAX_TENSOR_DIMS`; `XNNCompiler` load fail | [xnnpack-pixel-shuffle-load-failure.md](xnnpack-pixel-shuffle-load-failure.md) |
| `_softmax` | 33 | `XNNExecutor` shape propagation (`xnn_status_invalid_parameter`) | [xnnpack-softmax-skip.md](xnnpack-softmax-skip.md) |

Full SKIP/CRASH reason tables: [../skips.md](../skips.md). Ruled out: [../ruled_out/](../ruled_out/).
Coverage discussion (what injection changed; why exp/rsqrt/min/max show 0): [../README.md](../README.md).
