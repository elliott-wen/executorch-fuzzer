# Ruled-out suspects (QNN HTP emulator)

- **fp16 precision noise (~250)**: delta<0.05 mismatches (e.g. `rsqrt` 211) are the HTP fp16 float path
  vs the fp32 eager oracle — not bugs. The right reference for these is a fp16/quantized oracle
  (feeder/compare concern), not fp32 eager. Excluded.
- **INT64 sentinels (142)** and **portable-fallback mismatches (652, `ops=0`)** — int64 `2^63`
  confounds and ops that ran outside the HTP delegate; out of scope.
