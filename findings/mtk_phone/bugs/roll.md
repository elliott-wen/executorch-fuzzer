# `roll` — wrong element ordering (MISMATCH), delegated

- **Mode:** MISMATCH · **Bucket:** delegated · **Repro:** `repro_roll.py` · Deterministic 5/5, 187 samples.
- **Mechanism:** WRONG-VALUE (data movement). `roll` should cyclically shift; the device emits a wrong
  ordering and fills the tail with a repeated element. Example (fp32, shape `[2,3,2]`):
  - eager : `[-0.482, 1.110, 0.934, 1.164, 0.514, 0.675, 0.103, -0.137, ...]`
  - device: `[0.514, 0.934, -0.482, -0.279, -0.793, 0.103, -0.482, -0.482, -0.482, ...]`  (tail = -0.482 repeated)
- Affects fp16 and fp32. The shift index / wrap arithmetic in the op's lowering is wrong.
