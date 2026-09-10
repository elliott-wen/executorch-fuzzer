# Vulkan (Pixel 9, on-device) differential-fuzz — findings

Fifth on-device Vulkan run, and the **first non-Qualcomm GPU**: a **Pixel 9** (Google Tensor G4,
**ARM Mali** GPU), ExecuTorch **Vulkan** delegate, broker over `rathole`, diffed vs eager.
Compared 5-way against the four Adreno devices ([Galaxy](../vulkan_galaxy_26/README.md),
[ASUS](../vulkan_asus/README.md), [Xiaomi 13 Pro](../vulkan_xiaomi_13pro/README.md),
[Galaxy Tab S4](../vulkan_galaxytab_s4/README.md)). Synthesis:
[../vulkan_cross_device/README.md](../vulkan_cross_device/README.md).

Verdicts: **OK 13,459 / MISMATCH 8,072 / CRASH 6,214 / SKIP 72,286 / TIMEOUT 0** (clean run).

## TL;DR — a different GPU vendor, nearly identical behaviour

Despite being **Mali, not Adreno**, the Pixel 9 is almost a carbon copy of the Galaxy phone:

| | Galaxy (Adreno) | **Pixel 9 (Mali)** |
|---|---:|---:|
| OK | 13,331 | 13,459 |
| MISMATCH | 8,202 | 8,072 |
| CRASH | 6,214 | **6,214** |
| SKIP | 72,285 | **72,286** |

- **SKIP op surface — identical** (72,286 vs 72,285; **0 Pixel9-only skips** vs all others). The Mali
  driver has the same relevant Vulkan features (incl. `VK_KHR_8bit_storage`) → no Tab-S4-style
  collapse. Confirms the op surface is set by host lowering + the device feature set, **not the GPU
  vendor**.
- **CRASH — identical** (6,215; **0 Pixel9-only crashes**). The crash culprits (`unfold_copy`,
  `narrow_copy`, the copy/view family) are **GPU-vendor-invariant**.
- **MISMATCH — the device-invariant delegate bugs all reproduce.** Only **19 Pixel9-only** mismatches.

**Big result:** every device-invariant delegate bug (int64 truncation, fp16 overflow, NaN-dropping
activations, floor_divide, index_put, copy-elision aliasing) reproduces on Mali too — so they are
genuinely **delegate-level** (lowering/shaders), not Adreno-specific. Nothing new on correctness —
reuse [../vulkan_galaxy_26/README.md](../vulkan_galaxy_26/README.md).

## The one surprise: the transcendental fringe splits devices into camps, NOT by vendor

The numeric special-value fringe does **not** track GPU vendor. Pixel 9 (Mali) lands with **Galaxy
(Adreno)**, while ASUS and Xiaomi (both Adreno) form the other camp:

| pair | shared mismatches | the divergent side |
|---|---:|---|
| galaxy ∩ pixel9 | 8,051 | galaxy-only 151, **pixel9-only 21** (tiny) |
| asus ∩ pixel9 | 8,045 | **asus-only 851**, pixel9-only 27 |

ASUS's 851 extra are its `sqrt`/`rsqrt` transcendental non-finites
([../vulkan_asus/bugs/asus-transcendental-nonfinite.md](../vulkan_asus/bugs/asus-transcendental-nonfinite.md))
— **Pixel 9 does NOT have them** (it matches eager there, like Galaxy). So the two camps are:

- **"quiet" on transcendental special-values:** Galaxy (Adreno) + Pixel 9 (Mali)
- **"loud" (sqrt/rsqrt NaN/Inf fringe):** ASUS (Adreno) + Xiaomi (Adreno)

Since the split crosses the Adreno/Mali boundary, it's **not a hardware-architecture difference** —
it points to **driver version / precision mode (fast-math / RelaxedPrecision flushing)**, not the GPU
family. Two Adreno phones disagree; an Adreno and a Mali agree.

## Pixel 9's own tiny fringe (19 mismatches, vs all 4 others)

Not the sqrt/rsqrt class — top ops `floor_divide` (4), `sum` (2), then `logit`/`var`/`gather`/`cosh`/
`fmod`/`minimum` (1 each), only 3 non-finite. A small Mali-specific numeric fringe distinct from the
Adreno transcendental one; same character (special-value / rounding edges), different ops. Not a
delegate bug.

## Distribution & 5-way comparison
[../vulkan_cross_device/README.md](../vulkan_cross_device/README.md) — verdicts, 5-way agreement
(78.1%), pairwise overlaps, device camps. Raw data + script: `tmp/run_vulkan_pixel9/`
(`skip_reasons_pixel9.tsv`, `compare_5way.py` → `compare_5way.txt`).
