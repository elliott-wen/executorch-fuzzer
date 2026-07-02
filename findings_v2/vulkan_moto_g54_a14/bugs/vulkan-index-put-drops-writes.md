# Vulkan delegate: index_put does not write its values

**Signature:** `aten.index_put` returns the destination with the indexed writes absent; in the
`accumulate=False` path element 0 of the output is additionally 0.
**Classification:** real-bug — **kernel** (indexed writes not applied), not aliasing.
**Status:** **device-confirmed on the Moto G54 5G** in isolation, 2026-06-29.

## Phone-verified behavior

Single-op graphs, each run on the Moto. Exact eager (CPU) vs device outputs:

| input `self` | indices / values | accumulate | eager | **device** |
|---|---|---|---|---|
| `[1,2,3,4]` | `[1,3] ← [7,8]` | False | `[1,7,3,8]` | `[0,2,3,4]` |
| `[10,20,30,40]` | `[2] ← [99]` | False | `[10,20,99,40]` | `[0,20,30,40]` |
| `[0,0,0,0]` | `[1,3] ← [7,8]` | False | `[0,7,0,8]` | `[0,0,0,0]` |
| `[1,2,3,4]` | `[1,3] ← [7,8]` | True | `[1,9,3,12]` | `[1,2,3,4]` |

Reading the verified outputs:
- **accumulate=False:** device output equals the input `self`, except **element 0 is 0**; the
  written values (7, 8, 99) never appear. (With a zeros input this looks like all-zeros.)
- **accumulate=True:** device returns the input `self` **unchanged**; the writes are absent and
  element 0 is not zeroed.

In both paths the indexed values are not written.

## Repro
```
python mobile/findings_v2/vulkan_moto/bugs/repro_vulkan-index-put-drops-writes.py
```
Needs a connected Vulkan worker (broker on `127.0.0.1:15554`).

## Notes
- Vulkan-specific (portable/eager correct).
- Single-op isolation rules out the memory-planning / aliasing class — this is the kernel.
- `scatter.src_out` shows the same verified pattern (`vulkan-scatter-src-drops-writes.md`).
- Root cause in the Vulkan lowering is not pinned here; only the device input→output pairs above
  are asserted.
