# Ethos-U operator bug — `slice_scatter.default` (spurious NaN in scattered element)

**Failure mode:** mismatch (non-finite) · **Verdict:** NONFINITE_OUT, **5/5 stable** · **Severity: med**
**Device:** Arm Corstone-300 FVP (Ethos-U55, int8/Vela) — vs the PT2E-quantized CPU reference.

## Symptom (device-verified, `w1:149` out[0])
```
reference (CPU int8): [-3.0,    1.58415]
device    (Ethos-U):  [NaN,     1.58415]
input dtypes: float32, int64
```
`slice_scatter` writes the value `−3.0` into the scattered position; the device writes **NaN** there
instead. The other (non-scattered) element is correct.

## Root cause
Vela lowers `slice_scatter` to a **MemoryCopy/Slice** on the NPU (int8, scale ≈ 1.0, zp 0). Only the
**scattered element is wrong** (NaN) while the copied-through element is right, so the defect is in
the **scatter store path**: the element that should receive the sliced source value instead reads an
**uninitialised / mis-addressed** location and returns NaN. Inputs are finite, so the NaN is created
by the op. (`slice_scatter`'s int64 index input is also handled on the CPU path — a possible
interaction — but the failure is deterministic 5/5.)

## Reproduce
```
source tmp/ethos_env.sh
BISECT_BACKEND=ethos-u BISECT_CORPUS=corpus_v2/ethos-u BISECT_QUANTIZE=1 BROKER_PORT=15574 \
  ISO_DEV_TIMEOUT=200 python tmp/isolate_op.py w1:149 0
```
Stable `NONFINITE_OUT`, `eager=[-3.0, 1.584]` vs `device=[NaN, 1.584]`, 5/5.
