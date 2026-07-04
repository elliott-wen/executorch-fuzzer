# Confirmed Ethos-U operator bugs — index

All are **delegated** (Ethos-U NPU kernel, `delegated ops≥1`), on **finite** inputs, and
**deterministic** (REPRO 5/5 on the Corstone-300 FVP). Each has a runnable one-op repro.
Mechanisms derived from device-verified `eager` vs `device` value pairs.

## MISMATCH — WRONG-VALUE (7 operators)

| operator | mechanism | prevalence (delegated FINITE-CLEAN) | repro | doc |
|----------|-----------|------|-------|-----|
| `fill.Scalar` | fill value ignored; output = fixed input-derived garbage, same pattern for any fill value | 96/100 | `repro_fill-scalar.py` | [fill-scalar.md](fill-scalar.md) |
| `rsub.Scalar` | output collapses to ~0 past the first element(s) (dropped stores) | 50/257 | `repro_rsub-scalar.py` | [rsub-scalar.md](rsub-scalar.md) |
| `floor_divide` | rounding-direction off-by-one at integer boundary | 9/307 | `repro_floor-divide.py` | [floor-divide.md](floor-divide.md) |
| `bitwise_right_shift.Tensor_out` | logical (not arithmetic) shift of negative operands | 6 | `repro_bitwise-right-shift.py` | [bitwise-right-shift.md](bitwise-right-shift.md) |
| `clamp.Tensor_out` | min/max tensor broadcast fails; output = constant (first value) | 3/3 | `repro_clamp-tensor.py` | [clamp-tensor.md](clamp-tensor.md) |
| `sub.out` | scalar-broadcast subtraction wrong past first element | 1 | `repro_sub-out.py` | [sub-out.md](sub-out.md) |
| `remainder.Tensor_out` | wrong sign with negative divisor (C `%` vs floored) | 1 | `repro_remainder-tensor.py` | [remainder-tensor.md](remainder-tensor.md) |

Two recurring families: **broadcast failures** (`clamp.Tensor_out`, `sub.out`, and the tail-zeroing
in `rsub.Scalar`) and **integer sign/rounding semantics** (`bitwise_right_shift`, `remainder`,
`floor_divide`). `fill.Scalar` stands alone: the scalar operand is never applied.

## SKIP — coverage gaps (see ../skips.md)
`stack` and the constant-creation ops (`zeros/ones/full/arange`) fail to emit output on the FVP;
`bitwise_left/right_shift.Tensor_out` SKIP (rc=2) on many forms. Reasons tabulated in `../skips.md`.
