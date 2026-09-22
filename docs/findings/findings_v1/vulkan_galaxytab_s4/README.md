# Vulkan (Galaxy Tab S4, on-device) differential-fuzz — findings

Third on-device Vulkan run: `corpus/vulkan` (100,032 graphs) on a **Galaxy Tab S4**
(Adreno 630 / Snapdragon 835 — an older part than the two phones), ExecuTorch **Vulkan** delegate,
broker over `rathole`, diffed vs eager. Compared 3-way against
[Galaxy phone](../vulkan_galaxy_26/README.md) and [ASUS](../vulkan_asus/README.md).

## TL;DR — the Tab S4 is very different, for ONE reason

Unlike Galaxy-vs-ASUS (99.3% agreement), the Tab S4 agrees with the phones only **79.5%** — but it
is **not less buggy, it just runs far less**. A single missing GPU feature shrinks its op surface:

| verdict | Galaxy | ASUS | **Tab S4** |
|---|---:|---:|---:|
| OK | 13,331 | 12,637 | **1,488** |
| MISMATCH | 8,202 | 8,896 | **1,195** |
| CRASH | 6,214 | 6,214 | **4,692** |
| SKIP | 72,285 | 72,285 | **92,657** |

The Tab S4 **skips +20,396 more** graphs than the phones. **All of them trace to one cause** →

## The headline → [bugs/device-capability-8bit-storage.md](bugs/device-capability-8bit-storage.md)

**The Adreno 630 / SD835 Vulkan driver lacks `VK_KHR_8bit_storage`.** The delegate needs it for
every 8-bit (uint8/int8/bool) tensor: the input/output staging shaders
(`nchw_to_buffer_uint8_uint8`, `nchw_to_image_texture3d_uint8_uint8`) and every comparison-op output
(`binary_lt/gt/eq/le/ge_buffer_*` → bool). On the Tab S4 these are "**not compatible with device. …
VK_KHR_8bit_storage**" → **26,681** graphs refused (20,396 unique to the Tab S4). Bool is stored as
uint8, so any graph touching a comparison/logical result is rejected at the staging boundary.

This **overturns the "SKIPs are device-invariant" conclusion** from the ASUS comparison: that held
only because Galaxy and ASUS are similar modern parts (identical 72,285 skips). Across an Adreno
*generation*, the **device's Vulkan feature set defines the op surface**.

## The Tab S4 masks real bugs — it isn't more correct

Of its 20,396 extra skips, on Galaxy those same graphs were **11,774 OK, 7,090 MISMATCH, 1,532
CRASH**. So the Tab S4 hides 7,090 real mismatches and 1,532 crashes simply by refusing to run them.

## When the Tab S4 *does* run a graph, the bugs are identical

**1,102 of its 1,195 mismatches (92%) are shared by all three devices** — the delegate's correctness
bugs (int64 truncation, fp16 overflow, NaN-dropping activations, floor_divide, …) are
**device-invariant**; see [../vulkan_galaxy_26/README.md](../vulkan_galaxy_26/README.md). Only **17**
Tab S4-only mismatches (its small transcendental fringe — vs ASUS's 628). It crashes on the same ops
too (4,682 shared).

## Distribution & 3-way comparison

[00-distribution.md](00-distribution.md) — verdict table, 3-way agreement, pairwise overlaps, the
masking breakdown. Raw data + script: `tmp/run_vulkan_galaxytab_s4/`
(`skip_reasons_galaxytab_s4.tsv`, `compare_3way.py` → `compare_3way.txt`).
