# prod.int_out / sum.IntList_out — WRONG-VALUE (integer reduction overflow)

- **Failure mode:** MISMATCH (wrong value) · **delegated** · deterministic 5/5 · single-output, shapes match
- **Mechanism:** WRONG-VALUE — low-width integer reductions overflow differently on the CUDA kernel
  than on the CPU reference (accumulator width / wraparound point differs).

## Device-verified evidence
- `prod.int_out` (w0:226, uint8, shape [2,2,4,2]): eager `0`, **device `255`** — the product wraps to
  0 on CPU (an even factor drives it to 0 mod 256) but the device kernel yields 255. max|delta|=255.
- `sum.IntList_out` (w102:69, int16, scalar): eager `1`, **device `0`**. max|delta|=1.

Both are single-output `out=` ops whose user output shape matches eager, so this is a genuine value
divergence (not the multi-output alignment artifact that affects `min.dim`/`topk` — see ruled_out).

## Reproduce
```bash
... .venv-cuda/bin/python findings/cuda_h200/bugs/repro.py w0:226   # prod.int_out
... .venv-cuda/bin/python findings/cuda_h200/bugs/repro.py w102:69  # sum.IntList_out
```

## Note
Lower severity than bit-shift (small counts: 20 + 17), but a real, deterministic integer-semantics
divergence. Likely the AOTInductor reduction uses a different accumulate dtype than aten's CPU path.
