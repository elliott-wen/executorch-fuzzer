# `convolution` — wrong output layout (MISMATCH), delegated

- **Mode:** MISMATCH · **Bucket:** delegated · **Repro:** `repro_convolution.py` · Deterministic 5/5, 9 samples.
- **Mechanism:** WRONG-VALUE (layout/stride). The device output holds roughly the right magnitudes but
  in wrong spatial positions, with a leading `0.0`; max|delta| up to ~5.6. Example (shape `[3,12,12]`):
  - eager : `[0.728, -0.410, 1.970, 1.478, -0.059, 0.283, -0.625, 0.463, -2.223, ...]`
  - device: `[0.000, -0.411, 0.282, 1.971, -0.059, -2.223, 0.282, 0.463, 0.000, ...]`
- A layout (NCHW/NHWC) or output-stride confusion in the conv lowering.
