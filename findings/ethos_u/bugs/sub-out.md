# ethos-u `sub.out` — scalar-broadcast subtraction produces wrong values (WRONG-VALUE)

- **Failure mode:** MISMATCH
- **Mechanism:** WRONG-VALUE (broadcast) — `sub.out(L0, L1)` with a scalar `L0` broadcast against a
  larger `L1` yields wrong element values; the device does not evaluate `L0 − L1` correctly across
  the broadcast.
- **Bucket:** delegated (ops≥1) — Ethos-U delegate kernel.
- **Determinism:** REPRO 5/5. Domain: FINITE-CLEAN.
- **Prevalence:** 1 delegated FINITE-CLEAN sample (`w72:408`); most `sub.out` graphs run on portable
  (this is a rare delegated broadcast form). Confirmed deterministic.
- **Repro:** `bugs/repro_sub-out.py` (job `w72:408`)

## Device-verified evidence
`L0 = int32 [-3]` (shape `(1,1,1)`) broadcast against `L1` (shape `(1,1,2,4,3)`):

```
eager  = [12, 12, -23, 12, 2, 17, 7, 12, …]
device = [12, 15,  15, 15, 12, 12, 19, 12, …]   (int32)   max|delta| = 38
```

The first element is correct (`-3 − (-15) = 12`) but subsequent broadcast positions diverge
(e.g. reference `-23` vs device `15`), the same broadcast-handling failure family as
`clamp.Tensor_out`.

## Reproduce
```
python findings/ethos_u/bugs/repro_sub-out.py
```
