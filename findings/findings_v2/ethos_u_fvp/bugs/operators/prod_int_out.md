# Ethos-U operator bug — `prod.int_out` (uint8-range values overflow signed int8)

**Failure mode:** mismatch (wrong value) · **Verdict:** GENUINE, **5/5 stable** · **Severity: high**
**Device:** Arm Corstone-300 FVP (Ethos-U55, int8/Vela) — vs the PT2E-quantized CPU reference.

## Symptom (device-verified, `w10:850` out[1])
```
reference (CPU int8): [2.0, 250.0, 251.0, 250.0, 253.0]
device    (Ethos-U):  [2.0,   0.0,   0.0,   0.0,   0.0]     max|delta| = 253.0
```
Every element in the **128…255** range collapses to **0**; the one small value (2) is correct.

## Root cause — same as `pow.Scalar_out`
The output values are in the **uint8 range (0…255)** but Vela represents the output as **signed int8**
(range −128…127, scale ≈ 1.0, zero-point 0). Any value ≥ 128 is outside the signed int8 range and
overflows to **0**. The element-by-element split (small value survives, all ≥128 die) is the
signature of a **uint8-values-as-signed-int8 quantization bug** — the same root cause as
`pow_Scalar_out.md`. This is a **single underlying defect** (wrong output dtype/scale/zero-point
selection for uint8-range reductions), not two independent kernel bugs.

## Reproduce
```
source tmp/ethos_env.sh
BISECT_BACKEND=ethos-u BISECT_CORPUS=corpus_v2/ethos-u BISECT_QUANTIZE=1 BROKER_PORT=15574 \
  ISO_DEV_TIMEOUT=200 python tmp/isolate_op.py w10:850 1
```
Stable `GENUINE`, `eager=[2,250,251,250,253]` vs `device=[2,0,0,0,0]`, 5/5.
