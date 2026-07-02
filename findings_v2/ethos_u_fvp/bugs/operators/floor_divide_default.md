# Ethos-U operator bug — `floor_divide.default` (off-by-one rounding)

**Failure mode:** mismatch (wrong value) · **Verdict:** GENUINE, **5/5 stable** · **Severity: low–med**
**Device:** Arm Corstone-300 FVP (Ethos-U55, int8/Vela) — vs the PT2E-quantized CPU reference.

## Symptom (device-verified, `w14:141` out[0])
```
reference (CPU int8): [0.0, 0.0, -1.0, -1.0,  0.0]
device    (Ethos-U):  [0.0, 0.0, -1.0, -1.0, -1.0]   max|delta| = 1.0  (worst idx 4: 0 vs -1)
input dtypes: float16, int8
```
One element is **off by one**: the reference floors to `0`, the device floors to `−1`.

## Root cause
`floor_divide` rounds toward −∞. At a value that sits exactly on (or within int8 rounding of) an
integer boundary, the device's int8 quantized division + floor rounds the **wrong direction** for one
element — it takes the floor a step too far negative. Unlike the `pow`/`prod` overflow bug this is a
**rounding / boundary** error (Δ = 1, not a range overflow): the quantized reciprocal-multiply-then-
floor sequence in the Vela lowering disagrees with the reference floor at the tie point. The other
elements are correct, so it is input-value-specific (only bites near integer boundaries).

## Reproduce
```
source tmp/ethos_env.sh
BISECT_BACKEND=ethos-u BISECT_CORPUS=corpus_v2/ethos-u BISECT_QUANTIZE=1 BROKER_PORT=15574 \
  ISO_DEV_TIMEOUT=200 python tmp/isolate_op.py w14:141 0
```
Stable `GENUINE`, 5/5. Note `floor_divide.out` (a different overload) was **flaky (0/5)** and is *not*
a confirmed bug — only `.default` reproduces.
