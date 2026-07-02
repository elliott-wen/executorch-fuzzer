# Vulkan delegate: scatter.src_out does not write its source

**Signature:** `aten.scatter.src_out` returns the destination with the scattered writes absent and
element 0 of the output set to 0.
**Classification:** real-bug — **kernel** (scatter writes not applied), not aliasing.
**Status:** **device-confirmed on the Moto G54 5G** in isolation, 2026-06-29. Same pattern also
observed on the Pixel 9 ([../../vulkan_pixel9/ISOLATION_PROBES.md](../../vulkan_pixel9/ISOLATION_PROBES.md)).

## Phone-verified behavior

Single-op graphs run on the Moto. Exact eager (CPU) vs device outputs:

| input `self` | dim / index / src | eager | **device** |
|---|---|---|---|
| `[1,2,3,4]` | `dim 0, [0,2] ← [5,6]` | `[5,2,6,4]` | `[0,2,3,4]` |
| `[0,0,0,0]` | `dim 0, [0,2] ← [5,6]` | `[5,0,6,0]` | `[0,0,0,0]` |

Reading the verified outputs: device output equals the input `self`, except **element 0 is 0**;
the source values (5, 6) never appear. (With a zeros input this looks like all-zeros.) This is the
same verified pattern as `index_put` (`vulkan-index-put-drops-writes.md`).

## Repro
```
python mobile/findings_v2/vulkan_moto/bugs/repro_vulkan-scatter-src-drops-writes.py
```
Needs a connected Vulkan worker (broker on `127.0.0.1:15554`).

## Notes
- Vulkan-specific (portable/eager correct).
- Single-op isolation rules out the memory-planning / aliasing class — this is the kernel.
- Root cause in the Vulkan lowering is not pinned here; only the device input→output pairs above
  are asserted.
