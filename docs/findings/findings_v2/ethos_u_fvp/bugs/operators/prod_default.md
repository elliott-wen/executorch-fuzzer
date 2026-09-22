# Ethos-U operator bug — `prod.default` (spurious NaN)

**Failure mode:** mismatch (non-finite) · **Verdict:** NONFINITE_OUT, **5/5 stable** · **Severity: med**
**Device:** Arm Corstone-300 FVP (Ethos-U55, int8/Vela) — vs the PT2E-quantized CPU reference.

## Symptom (device-verified, `w40:731` out[0])
```
reference (CPU int8): [-0.0]
device    (Ethos-U):  [NaN]
input dtype: float16
```
The product reduction should be **0** (`−0.0`); the device returns **NaN** with finite inputs.

## Root cause
The int8 product-reduction lowering emits **NaN where the true result is zero**. A zero factor in the
product should force the whole product to 0; instead the device path produces a non-finite value —
consistent with a `0 × ∞` or a log-domain accumulation (`exp(Σ log|x|)`) that mishandles a zero
factor, or an uninitialised accumulator lane. Inputs are finite (`inputs_nonfinite=false`), so the
NaN is manufactured by the op itself, not propagated. This is distinct from the `prod.int_out`
overflow bug — here the failure is a **spurious NaN on a zero result**, not a range overflow.

## Reproduce
```
source tmp/ethos_env.sh
BISECT_BACKEND=ethos-u BISECT_CORPUS=corpus_v2/ethos-u BISECT_QUANTIZE=1 BROKER_PORT=15574 \
  ISO_DEV_TIMEOUT=200 python tmp/isolate_op.py w40:731 0
```
Stable `NONFINITE_OUT`, `eager=[-0.0]` vs `device=[NaN]`, 5/5.
