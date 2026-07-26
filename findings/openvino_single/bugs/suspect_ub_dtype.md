# SUSPECT — UB shift counts / dtype-form divergences (lower confidence)

These are delegated + REPRO, but the mechanism is a semantic/UB grey area rather than a clear
functional bug. Reported separately so they don't inflate the confirmed-bug count.

## `bitwise_left_shift.Tensor_out` / `bitwise_left_shift.Tensor_Scalar_out` — UB shift counts
```
job w103:452  bitwise_left_shift.Tensor_out(L0, L1)
leaf : L0=[-3,-3] int64, L1=[-3,4] int64   (shift by -3 is UB)
eager : [0.0, -48.0]        device: [-inf, -48.0]
```
Element 1 (`-3 << 4 = -48`) matches. Element 0 shifts by **-3** — a **negative shift count is
undefined behaviour** in C/C++; PyTorch yields `0`, OpenVINO yields `-inf`/`2^63`-scale garbage.
Large shift counts (≥ bit width) similarly diverge. Because the input is UB, this is a
semantics-difference, not a clear-cut kernel bug — **lower confidence**. (The delegated `.Tensor_out`
instances are here; the **portable-fallback** `bitwise_left_shift.Tensor_Scalar` / `.Tensor_out`
mismatches are ruled out per step 3a — see `ruled_out/`.)

## `prod.int_out` — dtype-form divergence
```
job w0:226  prod.int_out(L0, dim, keepdim, dtype=int16)
leaf : (2,4,3,1,4) fp16       eager: int16 [0,0,0,...]   device: fp16 [0,0,...,-2]
```
The reference casts the product to **int16** (small fp products round to 0); the device keeps
**fp16**. This is largely a **dtype-form / isolation artifact** (step 3c) — a quantized/typed `out=`
form where device-int-vs-reference dtype differs — rather than a proven value error. **Lower
confidence**; not promoted.
