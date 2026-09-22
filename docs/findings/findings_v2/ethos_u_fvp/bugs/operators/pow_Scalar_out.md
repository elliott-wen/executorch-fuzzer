# Ethos-U operator bug — `pow.Scalar_out` (uint8-range value overflows signed int8)

**Failure mode:** mismatch (wrong value) · **Verdict:** GENUINE, **5/5 stable** · **Severity: high**
**Device:** Arm Corstone-300 FVP (Ethos-U55, int8/Vela) — vs the PT2E-quantized CPU reference.

## Symptom (device-verified, `w15:459` out[2])
```
reference (CPU int8): [250.0]
device    (Ethos-U):  [0.0]        max|delta| = 250.0
input dtype: uint8
```
`pow` produces **250**, the device returns **0**.

## Root cause — from the Vela command stream
Vela lowers this output to a **signed int8** NPU tensor and clips it to the signed range:
```
IFM/OFM_PRECISION      : ACTIVATION_TYPE_SIGNED, ACTIVATION_PRECISION_B8   # int8
NPU_SET_OFM_ZERO_POINT : 0
NPU_SET_OFM_SCALE      : shift=31, scale=2147483649   # ≈ 1.0
NPU_SET_ACTIVATION_MIN : 0xFF80 = -128
NPU_SET_ACTIVATION_MAX : 0x007F =  127
```
The op's real output range is **0…255 (uint8)**, but Vela assigns **signed int8, scale ≈ 1.0,
zero-point 0**, which can only represent **−128…127**. 250 is outside that range, so the kernel (a
TOSA LUT/Table lookup indexed by the int8-reinterpreted input) reads the wrong (negative) index and
returns 0. A **quantization type / parameter-selection bug**: the output `(dtype, scale,
zero_point)` Vela picks cannot represent the operator's actual value range.

## Shared root cause with `prod.int_out`
`prod.int_out` shows the identical fingerprint — reference `[2, 250, 251, 250, 253]`, device
`[2, 0, 0, 0, 0]`: the small value (2) survives, every value ≥128 collapses to 0. **Same bug** —
uint8-range values as signed int8, everything ≥128 overflows to 0. See `prod_int_out.md`.

## Reproduce
```
source tmp/ethos_env.sh   # FVP + toolchain; broker on 127.0.0.1:15574
BISECT_BACKEND=ethos-u BISECT_CORPUS=corpus_v2/ethos-u BISECT_QUANTIZE=1 BROKER_PORT=15574 \
  ISO_DEV_TIMEOUT=200 python tmp/isolate_op.py w15:459 2
```
Stable `GENUINE`, `eager=[250.0]` vs `device=[0.0]`, 5/5.
