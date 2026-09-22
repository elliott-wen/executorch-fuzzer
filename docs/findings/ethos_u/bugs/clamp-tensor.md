# ethos-u `clamp.Tensor_out` — min/max tensor broadcast fails, output is constant (WRONG-VALUE)

- **Failure mode:** MISMATCH
- **Mechanism:** WRONG-VALUE (broadcast failure) — with tensor `min`/`max` operands that broadcast
  against a scalar input, the delegate applies only the **first** min/max element to the whole
  output, producing a constant tensor instead of the per-element clamp.
- **Bucket:** delegated (ops≥1) — Ethos-U delegate kernel.
- **Determinism:** REPRO 5/5. Domain: FINITE-CLEAN.
- **Prevalence:** 3 of 3 delegated FINITE-CLEAN `clamp.Tensor_out` samples MISMATCH.
- **Repro:** `bugs/repro_clamp-tensor.py` (job `w118:580`)

## Device-verified evidence
`n0 = clamp.Tensor_out(L0, min_t, max_t)` where `L0` is a single element broadcast to a large
output and `min_t`/`max_t` vary per element → the reference varies per element; the device returns
a **constant** equal to the first element:

| job | eager out (head)          | device out (head)      |
|-----|---------------------------|------------------------|
| w118:580 | `[13,0,14,13,11,7,1,7,…]` | `[13,13,13,13,13,13,13,13,…]` |
| w2:292   | `[15,1,15,2,13,14,2,14,…]`| `[15,15,15,15,15,15,15,15,…]` |
| w94:714  | `[0,0,0,8,1,3,1,11,…]`    | `[0,0,0,0,0,0,0,0,…]` |

The device output collapses to the first clamp result and never advances through the broadcast
`min`/`max` tensors. Output shape is correct; values are wrong.

## Reproduce
```
python findings/ethos_u/bugs/repro_clamp-tensor.py
```
