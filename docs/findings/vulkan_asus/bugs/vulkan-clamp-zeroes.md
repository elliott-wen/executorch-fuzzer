# Vulkan `clamp` (tensor bounds) — wrongly zeroes in-range values

- **Mode:** MISMATCH · **Root cause:** operator kernel (delegated) · **Occurrences:** 308 (`clamp.Tensor_out`)
- **Repro:** [repro_vulkan-clamp-mismatch.py](repro_vulkan-clamp-mismatch.py) (`w0:100`)

```
eager  : [-0.304932, 1.110352, ..., 1.49707]
device : [0.0,       1.110352, ..., 0.0]
```

With per-element tensor min/max bounds, Vulkan `clamp` drives some in-range values to `0.0` instead of
clamping to the bound — the tensor-bound broadcast/indexing is wrong.
