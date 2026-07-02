# SKIP reason: unimplemented argument paths

**Count: 11,141 (29.5% of skips).** The op exists and the dtype is fine, but a specific
**flag / mode / layout** is not implemented in the portable kernel, so it returns an error.
**These are the most actionable coverage gaps** — a real exported model could plausibly hit
them (unlike the dtype gaps or the malformed-input rejections).

## Clusters

| count | operator | firing file | unimplemented path |
|---:|---|---|---|
| 4,821 | `expand_copy.out` | `op_expand_copy.cpp` | `implicit == true` not implemented |
| 4,775 | `copy.out` | `op_copy.cpp` | `non_blocking == true` not supported |
| 1,119 | `_fft_r2c.out` | (reduce/fft) | `onesided == False` not supported yet |
| 375 | `_fft_r2c.out` | `tensor_util_portable.cpp` | inputs have different dim orders (no mixed contiguous/channels_last) |
| 51 | `_fft_r2c.out` | `reduce_util.cpp` | `onesided == False` not supported yet |

## Notes

- **`expand_copy implicit==true` (4,821).** The implicit-expand overload path is not wired
  up. ATen supports it; portable refuses.
- **`copy non_blocking==true` (4,775).** The `non_blocking` async-copy flag isn't supported
  (reasonable on a synchronous CPU runtime, but it errors rather than ignoring the flag —
  ATen treats `non_blocking` as a no-op hint on CPU and proceeds). Arguably portable should
  accept and ignore it rather than refuse.
- **`_fft_r2c onesided=False` (1,170).** The two-sided real FFT output path is explicitly
  "not supported yet" — a genuine feature gap that a spectral model would hit.
- **`_fft_r2c` mixed dim-orders (375).** Requires all inputs contiguous or all channels-last.

## Classification
Genuine feature gaps in the portable backend, not bugs (they fail gracefully). Priority for
"make portable run more real models": `copy non_blocking` (should be a no-op, not an error),
`expand_copy implicit`, and `_fft_r2c onesided`. These are the skip entries most worth wiring
up upstream.
