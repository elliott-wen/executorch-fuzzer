# ethos-u `remainder.Tensor_out` — wrong sign with negative divisor (WRONG-VALUE)

- **Failure mode:** MISMATCH
- **Mechanism:** WRONG-VALUE (sign handling) — `remainder` must follow the **divisor's** sign
  (Python/PyTorch semantics: `1 % -3 == -2`). The device computes the C `%` truncated remainder
  (`1`), giving the wrong sign for negative divisors.
- **Bucket:** delegated (ops≥1) — Ethos-U delegate kernel.
- **Determinism:** REPRO 5/5. Domain: FINITE-CLEAN.
- **Prevalence:** 1 delegated FINITE-CLEAN sample (`w70:394`); confirmed deterministic.
- **Repro:** `bugs/repro_remainder-tensor.py` (job `w70:394`)

## Device-verified evidence
`L0 = bool [False, True]`, `L1 = int8 [-3]`:

```
remainder(0, -3) : eager = 0    device = 0
remainder(1, -3) : eager = -2   device = 1     <-- wrong sign
```

`eager = [0,-2]`, `device = [0,1]` (int16). `1 % -3` is `-2` under PyTorch's floored remainder
(result takes the divisor's sign); the device returns `1`, the truncated C remainder.

## Reproduce
```
python findings/ethos_u/bugs/repro_remainder-tensor.py
```
