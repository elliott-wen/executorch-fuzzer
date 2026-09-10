# Vulkan (Xiaomi 13 Pro, on-device) differential-fuzz — findings

Fourth on-device Vulkan run: `corpus/vulkan` (100,032 graphs) on a **Xiaomi 13 Pro**
(Snapdragon 8 Gen 2 / **Adreno 740** — modern high-end), ExecuTorch **Vulkan** delegate, broker
over `rathole`, diffed vs eager. Compared 4-way against [Galaxy](../vulkan_galaxy_26/README.md),
[ASUS](../vulkan_asus/README.md), and [Galaxy Tab S4](../vulkan_galaxytab_s4/README.md).
Cross-device synthesis: [../vulkan_cross_device/README.md](../vulkan_cross_device/README.md).

> Note: the run was stopped with ~19 jobs unsent and **1,602 TIMEOUTs** (see below), so OK is a
> slight under-count. Verdicts captured: SKIP 71,033 / MISMATCH 8,733 / CRASH 6,244 / TIMEOUT 1,602
> / OK 12,420.

## TL;DR — a modern phone, behaves like ASUS/Galaxy

The Adreno 740 has `VK_KHR_8bit_storage`, so the Tab S4's capability collapse does **not** happen
here: **0 Xiaomi-only skips**, SKIP=71,033 (≈ the phones' 72,285). Its mismatches are near-identical
to ASUS:

| pair | shared mismatches | xiaomi-only |
|---|---:|---:|
| asus ∩ xiaomi | **8,732** | **1** |
| galaxy ∩ xiaomi | 8,043 | (690 = the transcendental fringe, like ASUS) |

So the Xiaomi reproduces every device-invariant delegate bug **and** the ASUS-style transcendental
fringe. **Just 1 truly Xiaomi-only mismatch.** Nothing new on the correctness front — reuse
[../vulkan_galaxy_26/README.md](../vulkan_galaxy_26/README.md).

## What IS Xiaomi-specific (both environmental / driver-edge, not delegate bugs)

### 1,602 TIMEOUTs — transient, not a kernel-hang bug
The only device with timeouts (others: 0). But the timeout graphs show **no op clustering** (max op
lift 1.36, vs 2.9× for a real crash culprit) — uniform across ~all ops. That signature = **transient
device/connection stalls** over the long ~2 h run (it also ran slower, ~14.5/s vs 34-50/s), not a
specific hung kernel. These are "unknown" verdicts; a re-feed would clear most. Not a finding about
the delegate.

### 139 Xiaomi-only crashes — driver aborts where others throw
Crashes unique to Xiaomi (CRASH on Xiaomi, not on the other three). On the other devices these are:
**99 SKIP, 23 OK, 17 MISMATCH.** The dominant 99 are graphs that **gracefully SKIP** on Galaxy/ASUS
(a caught `CppException`) but **native-abort** on the Xiaomi driver — top op `to` (99, the
dtype-conversion / staging path). I.e. on a staging/conversion edge case, the Adreno 740 driver
aborts instead of throwing a catchable error. Same **abort-vs-throw** class as the
[xnnpack-arm finding](../xnnpack-arm/README.md) (ARM proceeds and aborts where x86 errors early) —
device-specific *reachability* of a shared edge, not a new bug class. The remaining ~40 (OK/mismatch
elsewhere) are likely the same run instability as the timeouts.

## Distribution & 4-way comparison
[../vulkan_cross_device/README.md](../vulkan_cross_device/README.md) — verdicts, 4-way agreement
(78.1%), pairwise overlaps. Raw data + script: `tmp/run_vulkan_xiaomi_13pro/`
(`skip_reasons_xiaomi_13pro.tsv`, `compare_4way.py` → `compare_4way.txt`).
