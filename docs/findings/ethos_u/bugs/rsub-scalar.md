# ethos-u `rsub.Scalar` — output collapses to ~0 past the first elements (WRONG-VALUE / ZEROED)

- **Failure mode:** MISMATCH
- **Mechanism:** WRONG-VALUE / ZEROED — `rsub.Scalar(L0, other, alpha) = other − alpha·L0` is
  computed correctly for the first element(s) only; the remainder of the output tensor is dropped
  to (or near) zero, so large-magnitude reference values read as 0 on device.
- **Bucket:** delegated (ops≥1) — Ethos-U delegate kernel.
- **Determinism:** REPRO 5/5. Domain: FINITE-CLEAN.
- **Prevalence:** 50 of 257 delegated `rsub.Scalar` samples MISMATCH (FINITE-CLEAN); a further 39
  are non-finite-input (ruled out).
- **Repro:** `bugs/repro_rsub-scalar.py` (job `w100:292`)

## Device-verified evidence
The eager reference has full dynamic range; the device output matches the first element then
collapses toward zero:

| job | eager out (head)                              | device out (head)                       |
|-----|-----------------------------------------------|-----------------------------------------|
| w107:507 | `[-4.85, -5.85, -7.29, -6.96, -6.19, …]` | `[-4.85, 0, 0, 0, 0, …]`                |
| w116:372 | `[-3.84, -4.84, -6.29, -5.99, -5.17, …]` | `[-3.84, 0, -0.41, -0.21, -0.03, 0, …]` |
| w100:292 | `[0.16, -0.84, -2.31, -1.97, -1.20, …]`  | `[0.16, -0.63, 1.04, 0.97, 1.04, …]`    |

Max |delta| up to 8.5 on values whose int8 quantization step is ≈0.1, i.e. ~85× the quantization
noise floor — far beyond a near-tolerance artifact. The first output element is consistently
correct while the tail is zeroed/garbage, pointing at a dropped-store / partial-write in the
delegated kernel rather than a scale error.

## Reproduce
```
python findings/ethos_u/bugs/repro_rsub-scalar.py
```
