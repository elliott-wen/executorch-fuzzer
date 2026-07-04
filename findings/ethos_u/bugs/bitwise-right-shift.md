# ethos-u `bitwise_right_shift.Tensor_out` — logical instead of arithmetic shift on negatives (WRONG-VALUE)

- **Failure mode:** MISMATCH
- **Mechanism:** WRONG-VALUE (sign handling) — right-shifting a **negative** operand by a
  non-negative amount uses a logical / round-toward-zero shift on device, where the reference
  (PyTorch) uses an arithmetic shift (round toward −∞, sign-extending).
- **Bucket:** delegated (ops≥1) — Ethos-U delegate kernel.
- **Determinism:** REPRO 5/5. Domain: FINITE-CLEAN (shift amounts are **non-negative** — this is
  the well-defined case, distinct from the negative-shift UB samples in `ruled_out/`).
- **Prevalence:** 6 delegated FINITE-CLEAN (non-negative-shift) MISMATCH; the op also SKIPs on 86
  other delegated forms (see `skips.md`) and MISMATCHes on 400 negative-shift UB forms (ruled out).
- **Repro:** `bugs/repro_bitwise-right-shift.py` (job `w116:400`)

## Device-verified evidence
`L0 = int64 [-3,-3]`, shift `L1 = uint8 [1,4]` (both non-negative):

```
-3 >> 1 : eager = -2   device = -1     (arithmetic floor(-3/2)=-2  vs  trunc/logical -1)
-3 >> 4 : eager = -1   device =  0     (arithmetic -1              vs  logical 0)
```

`eager = [-2,-1]`, `device = [-1,0]`. The device fails to sign-extend the negative operand,
matching a logical (unsigned) right shift rather than the arithmetic shift PyTorch specifies.

## Reproduce
```
python findings/ethos_u/bugs/repro_bitwise-right-shift.py
```
