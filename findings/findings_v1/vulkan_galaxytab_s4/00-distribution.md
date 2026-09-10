# Vulkan 3-way device comparison — Galaxy vs ASUS vs Galaxy Tab S4

Same `corpus/vulkan` (100,032 graphs), same `.pte` builds, three Adreno devices.
Generator: `tmp/run_vulkan_galaxytab_s4/compare_3way.py` → `compare_3way.txt`.

## Verdicts

| verdict | galaxy | asus | tab_s4 |
|---|---:|---:|---:|
| OK | 13,331 | 12,637 | 1,488 |
| MISMATCH | 8,202 | 8,896 | 1,195 |
| CRASH | 6,214 | 6,214 | 4,692 |
| SKIP | 72,285 | 72,285 | 92,657 |

**3-way verdict agreement: 79,489 / 100,032 = 79.5%** — the drop from the 99.3% Galaxy-vs-ASUS
agreement is entirely the Tab S4's extra skips.

## SKIP — device-invariant between the phones, NOT across generations

| | galaxy | asus | tab_s4 | shared by all 3 |
|---|---:|---:|---:|---:|
| SKIP | 72,285 | 72,285 | 92,657 | 72,261 |
| device-ONLY (vs both others) | 0 | 0 | **20,396** | |

Galaxy and ASUS are byte-identical (0 device-specific). The Tab S4 adds **20,396** unique skips, all
from missing `VK_KHR_8bit_storage` → [bugs/device-capability-8bit-storage.md](bugs/device-capability-8bit-storage.md).
These 20,396 graphs were, on Galaxy: **11,774 OK, 7,090 MISMATCH, 1,532 CRASH** — i.e. the Tab S4
masks 7,090 real mismatches and 1,532 crashes by refusing to run them.

## CRASH — same ops, fewer (because more are skipped)

| | galaxy | asus | tab_s4 | shared by all 3 |
|---|---:|---:|---:|---:|
| CRASH | 6,214 | 6,214 | 4,692 | 4,682 |
| device-ONLY | 1 | 1 | 10 | |

The Tab S4 crashes less only because ~1,532 would-be crashes are pre-empted by the 8bit_storage
skip. The 4,682 shared crashes confirm crash culprits are device-invariant.

## MISMATCH — the delegate bugs are device-invariant; fringes are per-device

| pair | shared | left-only | right-only |
|---|---:|---:|---:|
| galaxy ∩ asus | 8,192 | galaxy 10 | asus 704 |
| galaxy ∩ tab_s4 | 1,102 | galaxy 7,100¹ | tab_s4 93 |
| asus ∩ tab_s4 | 1,178 | asus 7,718¹ | tab_s4 17 |

¹ the large "galaxy/asus-only vs tab_s4" counts are **mostly Tab-S4-skipped graphs**, not graphs the
Tab S4 ran correctly — see the 8bit_storage masking above.

- **MISMATCH shared by ALL 3 devices: 1,102** — the device-invariant delegate bugs (int64, fp16,
  NaN-dropping activations, floor_divide, …; [../vulkan_galaxy_26/README.md](../vulkan_galaxy_26/README.md)).
  This equals 92% of the Tab S4's 1,195 mismatches — when it runs, it reproduces the same bugs.
- **Device-specific fringe (mismatch on one device, not the other two):** ASUS **628** (511
  non-finite, 387 transcendental ops — `sqrt`/`rsqrt`/`sinh`/…), Tab S4 **17**, Galaxy **10**. ASUS
  is the transcendental outlier; the Tab S4's fringe is tiny because it runs so few transcendental
  graphs.

## Reading

Two distinct kinds of device dependence:
1. **Feature-set (capability) dependence — large.** The Tab S4's missing `VK_KHR_8bit_storage`
   removes ~20k graphs from its op surface. The effective surface of the Vulkan delegate is set by
   the device's Vulkan features, not just the host build.
2. **Numeric (special-value) dependence — small.** Among graphs all devices *can* run, only the
   transcendental NaN/Inf fringe differs (ASUS ~0.7%); every real delegate bug reproduces on all
   three.
