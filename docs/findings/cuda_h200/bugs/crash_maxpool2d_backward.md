# max_pool2d_with_indices_backward.grad_input — CRASH (device-side assert, index-OOB)

- **Failure mode:** CRASH native abort (243) + SKIP catchable assert (38) · **delegated** · deterministic
- **Captured native error** (reproduced directly in `.venv-cuda`):
  ```
  torchinductor_.../coj55...py:27: Assertion `index out of bounds: 0 <= tmp0 < 6` failed.
  ```
- **Mechanism:** the backward scatters gradients into `grad_input` using the `indices` input
  (the argmax positions from the forward max-pool). The corpus feeds an **arbitrary `indices`
  tensor**; when an index is out of range the generated Triton kernel hits a device-side
  `assert`, which aborts the CUDA context asynchronously → native abort (or, if caught first, a
  `0x1` SKIP). There is **no input validation** — the op should reject out-of-range indices with a
  catchable error instead of a device abort that takes the process down.

## Why crash vs skip
Same root cause; the split (243 CRASH / 38 SKIP) is a race between the async assert aborting the
context and the runtime catching the launch error — both trace to the same missing guard.

## Reproduce
```bash
CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64 \
  PYTHONPATH=/data/jwen929 /data/jwen929/mobile/.venv-cuda/bin/python \
  /data/jwen929/mobile/findings/cuda_h200/bugs/repro.py w0:625
```
Prints the `index out of bounds` assert and aborts (expected).

## Note
This trigger (out-of-range `indices`) is partly a fuzzer artifact — real max-pool backward gets
valid indices from the forward. The **bug** is the un-guarded device abort, not the wrong answer:
a hostile/incorrect `indices` should fail safely, not crash the runtime.
