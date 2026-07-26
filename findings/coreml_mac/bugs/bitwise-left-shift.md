# CoreML `bitwise_left_shift.Tensor_out` — integer shift overflow yields garbage / non-finite

- **Failure mode:** MISMATCH
- **Root cause:** operator kernel (delegated to Core ML; `delegated.ops=1`, `non_delegated=0`)
- **Mechanism:** WRONG-VALUE / NONFINITE — a large shift amount overflows and the device emits a
  wrong integer (`-2³¹`) or even a floating `-inf` for an integer op
- **Occurrences:** 20 delegated mismatches (the remaining 123+199 `bitwise_left_shift` mismatches are
  **portable-fallback**, `ops=0`, ruled out — see [../ruled_out/README.md](../ruled_out/README.md)).
  Device-verified, deterministic (5/5).
- **Repro:** [repro_bitwise-left-shift.py](repro_bitwise-left-shift.py) (jobs `w1:321`, `w100:28`)

## What happens
For shift amounts at/above the type width, the delegate produces values that disagree with the
reference — and in the int32 case emits a **non-finite** result for a pure-integer operation:

```
job w1:321   bitwise_left_shift(L0,L1)   int64, finite leaves
  eager : [0.0, 48.0]
  device: [-2147483648.0, 48.0]      # -2^31: shift done in 32-bit, overflowed

job w100:28  bitwise_left_shift(L0,L1)   int32, finite leaves
  eager : [0, 0, 0, 0, -6, 0, 8, 0]
  device: [-inf, -inf, -inf, -inf, -6, -inf, 8, -inf]   # -inf from an integer op
```

## Why
Two related lowering faults: (1) an int64 shift is evaluated in 32-bit, so `1 << k` overflows to
`-2³¹` instead of the correct 64-bit value / the reference's wrap-to-0; (2) the int32 path routes the
shift through a float representation, so an out-of-range shift becomes `-inf`. An integer bitwise op
must never yield `-inf`; the absence of a width/domain guard is the bug. (Left-shift past the width is
UB in C, so "correct" here means matching the reference wrap, but emitting `-inf` is unambiguously
wrong.)
