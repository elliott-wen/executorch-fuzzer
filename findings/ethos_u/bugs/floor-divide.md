# ethos-u `floor_divide` — rounding-direction off-by-one (WRONG-VALUE)

- **Failure mode:** MISMATCH
- **Mechanism:** WRONG-VALUE (rounding) — `floor_divide` rounds the quotient toward −∞ (floor); at
  boundary quotients the device rounds the other way, so `floor(1.16/1)=1` reads as `0` on device.
- **Bucket:** delegated (ops≥1) — Ethos-U delegate kernel.
- **Determinism:** REPRO 5/5. Domain: FINITE-CLEAN.
- **Prevalence:** 9 delegated FINITE-CLEAN `floor_divide` MISMATCH; a further 56 are
  non-finite-input (ruled out).
- **Repro:** `bugs/repro_floor-divide.py` (job `w100:380`)

## Device-verified evidence
`L0 = float16`, `L1 = int16`, broadcast:

```
eager  = [0, 0, -1.003, 0, -1.003, -1.003, -1.003, …, 0.9956]
device = [0, 0, -1.003, 0, -1.003, -1.003, -1.003, …, 0.0   ]   max|delta| = 0.996
```

Most elements agree; the diverging element is `floor_divide(1.164, 1) = floor(1.164) = 1`
(quantized ≈ 0.9956) which the device returns as `0` — the quotient is truncated/rounded toward
zero at the integer boundary instead of floored.

## Reproduce
```
python findings/ethos_u/bugs/repro_floor-divide.py
```
