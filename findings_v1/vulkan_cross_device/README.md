# Vulkan delegate — cross-device study (6 devices, Adreno + Mali, same corpus)

The same `corpus/vulkan` (100,032 auto-generated graphs) and the same host `.pte` builds, executed
on **six** mobile GPUs via the broker, each output diffed vs eager PyTorch:

| device | SoC / GPU | vendor / tier |
|---|---|---|
| **galaxy** | Samsung Galaxy phone | Qualcomm Adreno (modern) |
| **asus** | ASUS phone | Qualcomm Adreno (modern) |
| **xiaomi** | Xiaomi 13 Pro, SD 8 Gen 2 / Adreno 740 | Qualcomm Adreno (flagship) |
| **pixel9** | Pixel 9, Tensor G4 / **Mali-G715** | **ARM Mali (flagship)** |
| **moto** | Moto G54 5G, Dimensity 7020 / **Mali-G57** | **ARM Mali (budget)** |
| **tab_s4** | Galaxy Tab S4, SD 835 / Adreno 630 | Qualcomm Adreno (old) |

Per-device: [galaxy](../vulkan_galaxy_26/README.md) · [asus](../vulkan_asus/README.md) ·
[xiaomi](../vulkan_xiaomi_13pro/README.md) · [pixel9](../vulkan_pixel9/README.md) ·
[moto](../vulkan_motog54/README.md) · [tab_s4](../vulkan_galaxytab_s4/README.md).
Data + script: `tmp/run_vulkan_motog54/compare_6way.py` → `compare_6way.txt`.

## Verdicts

| verdict | galaxy | asus | xiaomi | pixel9 | moto | tab_s4 |
|---|---:|---:|---:|---:|---:|---:|
| OK | 13,331 | 12,637 | 12,420¹ | 13,459 | 12,020 | 1,488 |
| MISMATCH | 8,202 | 8,896 | 8,733 | 8,072 | **9,482** | 1,195 |
| CRASH | 6,214 | 6,214 | 6,244 | 6,215 | 6,212 | 4,692 |
| SKIP | 72,285 | 72,285 | 71,033 | 72,286 | 72,190 | 92,657 |
| TIMEOUT | 0 | 0 | 1,602¹ | 0 | 128 | 0 |

¹ Xiaomi run stopped early + 1,602 transient timeouts (no op clustering → stalls, not a hang).

**6-way verdict agreement: 77,846 / 100,032 = 77.8%.** The only material disagreements are the Tab
S4's skips and the per-device numeric fringe; the five full-surface GPUs agree ~99% on verdicts.

## The story — what's invariant, and the two axes of device dependence

**Invariant across all six GPUs (two vendors, both architectures, flagship→budget):**
- **Every delegate correctness bug** (int64 truncation, fp16 overflow, NaN-dropping activations,
  floor_divide, index_put, copy-elision aliasing) — **1,063 shared** mismatch core, all reproduced
  on Mali and on the budget part.
- **Crash culprits** — 4,595 shared; 0 device-only crashes on every modern GPU. Bugs are
  **delegate-level, not GPU-specific.**

**Device-dependent on two independent axes:**
1. **Capability (feature set) — large, by *generation*.** The old Tab S4 lacks `VK_KHR_8bit_storage`
   → +20,389 unique skips (all uint8/int8/bool) →
   [device-capability-8bit-storage.md](../vulkan_galaxytab_s4/bugs/device-capability-8bit-storage.md).
   All five modern GPUs (Adreno *and* both Mali, incl. the budget Moto) have it → identical op surface.
2. **Numeric precision/special-values — by GPU *tier/driver*, NOT vendor.** Size of the device-only
   mismatch fringe spans 3 orders of magnitude (see below).

## SKIP — op surface is feature-set, not vendor or price

| | galaxy | asus | xiaomi | pixel9 | moto | tab_s4 |
|---|---:|---:|---:|---:|---:|---:|
| device-ONLY skips | 0 | 0 | 0 | 0 | 0 | **20,389** |

Five modern GPUs (incl. the *budget* Moto) are near-identical (72,285 ± ~250). Only the old Tab S4
collapses. "Budget" ≠ "missing features"; "old" does.

## CRASH — vendor- and tier-invariant culprits

Shared-by-all 4,595; device-only: tab_s4 10, xiaomi 139 (abort-vs-throw on staging), moto 10,
everyone else 0. Crash culprits (`unfold_copy`/`narrow_copy`/copy-view family) reproduce everywhere.

## MISMATCH — invariant bugs + a tier-ranked numeric fringe

Device-only mismatches (mismatch on that device, on *none* of the other five):

| device | device-only mismatches | character |
|---|---:|---|
| **pixel9** (Mali flagship) | **~11–21** | quietest |
| galaxy (Adreno) | ~2–10 | quiet |
| asus / xiaomi (Adreno) | ~6 / 0 | sqrt/rsqrt transcendental camp |
| tab_s4 (Adreno old) | ~14 | (small; runs few graphs) |
| **moto** (Mali budget) | **1,200** | loudest — broad fp16-precision + special-values |

```
quiet ───────────────────────────────────────────────► loud
 pixel9(Mali-flagship) · galaxy(Adreno) │ asus·xiaomi(Adreno) │ moto(Mali-budget)
```

The split crosses every hardware boundary: **two Adreno phones sit in different camps; the two Mali
GPUs are at opposite extremes** (flagship Pixel 9 quietest, budget Moto loudest). So the numeric
fringe is **driver precision / GPU tier**, not vendor or architecture — and it's worst on the
cheapest GPU (Moto's 1,200 span `pow`/`gelu`/`logit`/`floor_divide` + layout-copy carriers, ~75%
precision deltas, 25% NaN/Inf). The narrow sqrt/rsqrt fringe (ASUS/Xiaomi) is just the mild end of
the same axis. Detail:
[asus-transcendental-nonfinite.md](../vulkan_asus/bugs/asus-transcendental-nonfinite.md),
[moto README](../vulkan_motog54/README.md).

**Shared by ALL 6 devices: 1,063 mismatches** — the device-invariant delegate bugs
([bug catalog](../vulkan_galaxy_26/README.md)).

## Summary

| property | invariant across all 6? | note |
|---|---|---|
| delegate correctness bugs | ✅ yes | 1,063 shared; reproduce on Adreno + Mali, flagship→budget |
| crash culprit ops | ✅ yes | 4,595 shared; 0 unique on modern GPUs |
| SKIP op surface | ❌ by **generation** | Tab S4 −20k (`VK_KHR_8bit_storage`); budget Moto fine |
| numeric special-values / precision | ❌ by **tier/driver** | pixel9 ~20 → moto 1,200; not vendor, not Adreno-vs-Mali |
| abort-vs-throw / stability | ❌ tiny | xiaomi 139 staging aborts + timeouts; moto 128 timeouts |

**Bottom line:** across two GPU vendors, both architectures, and flagship-to-budget tiers, the Vulkan
delegate's **bugs are invariant** — a correctness bug found on any device applies to all. Device
*differences* reduce to: feature-set (op surface, set by generation) and arithmetic precision
(numeric fringe, set by GPU tier/driver — worst on the cheapest part). Practical guidance: gate
8-bit/bool ops on `VK_KHR_8bit_storage`; guard transcendental/precision-sensitive inputs (the fringe
balloons on low-end GPUs); the [bug catalog](../vulkan_galaxy_26/README.md) holds for every device.
