# Ruled-out (QNN phone)

- **fp16 precision**: small-delta (Δ<0.05) mismatches, e.g. `rsqrt` (233) — HTP fp16 vs fp32 eager
  oracle, not bugs.
- **INT64 `2^63` confounds**: `bitwise_left_shift.*` mismatches are dominated by INT64_MAX sentinels
  + shift-UB (the genuine `bitwise_or`/`xor` value bug is filed separately).
- **portable-fallback** (`ops=0`) mismatches — ran outside the HTP delegate.
