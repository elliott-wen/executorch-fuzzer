# clamp.Tensor_out — CRASH (illegal memory access, GPU OOB)

- **Failure mode:** CRASH native abort (177) · **delegated** · deterministic
- **Captured native error** (reproduced directly in `.venv-cuda`):
  ```
  [cuda_allocator.cpp:248] cudaMemcpyAsync failed: an illegal memory access was encountered (48 bytes)
  [storage.h:158] In function memcpy_async(), assert failed (err == ...Ok): memcpy_async failed
  ```
- **Op form:** `clamp.Tensor_out(L0, min, max, out)` — clamp with **tensor** `min`/`max` bounds.
- **Mechanism:** an illegal GPU memory access during the op / its output copy — a broadcasting form
  of tensor-bounded clamp drives an out-of-bounds read or write on the device. Unlike the max-pool
  case there is no clean `index out of bounds` assert; the failure surfaces as a raw illegal memory
  access in the allocator's `cudaMemcpyAsync`, i.e. a genuine memory-safety bug in the kernel or its
  AOTI output-copy for this shape/broadcast combination.

## Reproduce
```bash
CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64 \
  PYTHONPATH=/data/jwen929 /data/jwen929/mobile/.venv-cuda/bin/python \
  /data/jwen929/mobile/findings/cuda_h200/bugs/repro.py w0:168
```

## Severity
High: an **illegal memory access** is a memory-safety fault (undefined behaviour, potential
corruption), not merely a wrong value. 177/436 sampled `clamp.Tensor_out` forms crash.
