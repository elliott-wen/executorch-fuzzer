# bitwise_left_shift / bitwise_right_shift (.Tensor_Scalar, .out) — WRONG-VALUE

- **Failure mode:** MISMATCH (wrong value) · **delegated** (`ops=1`) · deterministic 5/5
- **Mechanism:** WRONG-VALUE — a shift amount `≥ bit-width` (or negative) is undefined behaviour;
  the CUDA/Triton kernel emits garbage where PyTorch CPU gives a defined saturated result.
- **Ops / count:** `bitwise_right_shift.Tensor_Scalar` (211) + `.out` (83), `bitwise_left_shift.Tensor_Scalar` (202) + `.out` (72) — ~570 mismatches.

## Device-verified evidence (w0:196, `bitwise_right_shift.Tensor_Scalar`, int64)

```
eager  (CPU): [-1, -1, -1,  0, -1, -1, -1, -1]     # arithmetic right shift saturates
device (GPU): [7.2055e16, 1.3422e10, 7.2055e16, 1.3422e10, ...]   # garbage
max|delta| = 7.21e16   (rtol=0 atol=0)
```

`w0:237` (`bitwise_left_shift.Tensor_Scalar`, int64): eager `-1`, device `1.342e10`, max|delta|=1.34e10.

The device values are the operand shifted by `s mod 64`-style hardware behaviour (PTX `shl/shr`
wrap the shift count), whereas PyTorch CPU defines out-of-range shifts to saturate to `0`/`-1`. The
kernel does not clamp the shift amount to the type width.

## Reproduce
```bash
CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64 \
  PYTHONPATH=/data/jwen929 /data/jwen929/mobile/.venv-cuda/bin/python \
  /data/jwen929/mobile/findings/cuda_h200/bugs/repro.py w0:196
```

## Fix direction
Clamp/guard the shift amount to `[0, bitwidth)` in the shift lowering (match aten CPU semantics),
or mark out-of-range shifts unsupported at partition time.
