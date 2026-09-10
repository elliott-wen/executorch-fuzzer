# Galaxy Tab S4: missing `VK_KHR_8bit_storage` collapses the op surface

**Signature:** `executor CppException: Shader <name> not compatible with device. Missing support for
extension or physical device feature: VK_KHR_8bit_storage`
**Count:** 26,681 SKIPs on the Tab S4 (20,396 unique vs the phones).
**Classification:** **device-capability gap** (hardware/driver), not a delegate bug.

## What happens

The Galaxy Tab S4 (Adreno 630 / Snapdragon 835) Vulkan driver does **not** expose
`VK_KHR_8bit_storage`. The ExecuTorch Vulkan delegate requires it for any tensor stored in 8 bits —
**uint8, int8, and bool** (bool is materialized as uint8). When such a tensor must cross the
host↔GPU staging boundary or be produced by a comparison op, the required shader is rejected at load
time and the whole graph is refused.

Every one of the 26,681 failures names the same missing feature:

| failing shader | n | role |
|---|---:|---|
| `nchw_to_buffer_uint8_uint8` | 18,508 | host→GPU **input staging** of a uint8/bool tensor |
| `nchw_to_image_texture3d_uint8_uint8` | 4,902 | input staging into a texture |
| `binary_{lt,gt,eq,le,ge}_buffer_{float,int32}` | ~2,500 | **comparison ops** (output is bool=uint8) |
| `image_to_nchw_… / clone_image_to_buffer_… / pad_buffer_uint8` | ~30 | output staging / misc uint8 |

`grep 'Missing support'` over the Tab S4 skip log → **100% `VK_KHR_8bit_storage`** (26,681/26,681).

## Why the phones don't hit it

Galaxy and ASUS (newer Adreno + driver) expose `VK_KHR_8bit_storage`, so their uint8/bool staging
shaders load fine. Their uint8 skips are a *different, smaller* class — the missing
`view_convert_buffer_*→uint8` **conversion** shader (~11k, a shader-gen gap, see
[../../vulkan_galaxy_26/skip/README.md](../../vulkan_galaxy_26/skip/README.md)). So:

- **Phones:** bool/uint8 *outputs* skip (no convert-out shader) — but bool *internal* tensors run.
- **Tab S4:** *any* 8-bit tensor crossing staging or produced by a comparison skips (no 8bit_storage)
  — a much larger set, hence +20,396.

## Impact

The Tab S4 runs only ~7,375 graphs (vs ~27,700 on the phones). Its lower MISMATCH/CRASH counts are an
artifact of this masking, not better correctness: the 20,396 extra skips were **11,774 OK / 7,090
MISMATCH / 1,532 CRASH** on Galaxy. When the Tab S4 *does* run a graph, 92% of its mismatches are
shared by all three devices — the delegate bugs are device-invariant.

## Takeaway

The **device's Vulkan feature set defines the effective op surface** of the delegate. A model using
bool/comparison/quantized(int8) ops that runs on a modern Adreno phone may be **entirely unsupported**
on an Adreno-630-class part for want of `VK_KHR_8bit_storage`. Detect the feature at lowering time and
either fall back (8-bit→int32 emulation) or fail loudly, rather than refusing graph-by-graph at
runtime.

## Repro / verify
```
grep -c 'VK_KHR_8bit_storage' tmp/run_vulkan_galaxytab_s4/skip_reasons_galaxytab_s4.tsv   # 26681
grep -c 'VK_KHR_8bit_storage' tmp/run_vulkan_galaxy_26/skip_reasons_mobile.tsv            # 0
```
