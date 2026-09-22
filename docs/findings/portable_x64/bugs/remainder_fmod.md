# `remainder` / `fmod` — portable kernel returns wrong sign/value vs ATen

- **Mode:** MISMATCH (finite values) · **Occurrences:** `remainder.Tensor_out` 44, `remainder.Scalar`
  39, `remainder.Scalar_out` 16, `fmod.Scalar` 13 · **Repro:** [repro_remainder.py](repro_remainder.py) (job `w1:333`)

```
eager (ATen)    : [0.0, -2.0]
device (portable): [0.0, 1.0]
```

The portable `remainder` disagrees with ATen on the result **sign/magnitude**. ATen `remainder`
follows the divisor's sign (result `-2`); the portable kernel returns `1`. `fmod` shows the same
family (fmod follows the dividend's sign — the two rounding conventions are being conflated). These
are finite-input value bugs, not a non-finite artifact.
