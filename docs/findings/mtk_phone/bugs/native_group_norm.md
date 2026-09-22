# `native_group_norm` — wrong normalized values (MISMATCH), delegated

- **Mode:** MISMATCH · **Bucket:** delegated · **Repro:** `repro_native_group_norm.py` · Deterministic 5/5, 97 value samples.
- **Mechanism:** WRONG-VALUE. fp32, finite input & finite eager reference; device normalization diverges
  well beyond fp16 tolerance (max|delta| ~0.4–1.0 on unit-scale outputs). Example (shape `[3,2]`):
  - eager : `[-1.407, 0.827, 0.580, 1.373, -0.979, -0.395]`
  - device: `[-2.039, 0.823, 0.507, 0.920, -0.249, 0.041]`
- Likely a group-statistics (mean/var) or affine error in the decomposed lowering.
- **NB:** group_norm also has 176 *non-finite* "mismatches" that are RULED OUT — the eager reference
  there is degenerate all-NaN, not a device bug. See `../ruled_out/nonfinite-reference-confound.md`.
