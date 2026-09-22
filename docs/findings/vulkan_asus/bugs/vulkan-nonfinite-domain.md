# Vulkan non-finite domain bugs (`gelu`, `mean`, `floor_divide`, `sqrt`, `rsqrt`, `addmm`)

- **Mode:** MISMATCH (non-finite) · **Root cause:** operator kernels (delegated) · surfaced by leaf injection
- **Repros:** [repro_vulkan-mean-mismatch.py](repro_vulkan-mean-mismatch.py), [repro_vulkan-floor_divide-mismatch.py](repro_vulkan-floor_divide-mismatch.py)

Device-verified on the ASUS phone:

| op | count | eager → device | note |
|---|---:|---|---|
| `gelu` | 371 | `gelu(inf)`: NaN → **inf** | shared with x86/arm64 xnnpack |
| `mean` | 139 | overflow: inf → **65504** | saturates to fp16-max instead of inf |
| `floor_divide` | 430 | `x//0`: NaN → **inf** | Vulkan returns inf on div-by-zero |
| `sqrt` | 40 | non-finite domain wrong | |
| `rsqrt` | 29 | non-finite domain wrong | |
| `addmm` | 24 | non-finite input → wrong | |

These fire on non-finite inputs (from the corpus injection) or op-produced overflow; the Vulkan
kernels lack the IEEE non-finite handling ATen has.
