# Vulkan operator bug — `asin.out`

**Failure mode:** mismatch  ·  **Root cause:** operator — **NON-DETERMINISTIC kernel**
**Divergence type:** wrong value  ·  **Device:** ASUS a12201 (Vulkan) — device-verified
**Stability:** NON-DETERMINISTIC — wrong on 1/7, correct on 6/7 of the SAME input

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w105:732` `out[1]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] dtype torch.float32 vs torch.int64`
- **worst element** (idx 1): eager **-1.5708** vs device **-9.223372036854776e+18**
- input dtypes: ['torch.int32']
- sample idx [0, 1] — eager  [-1.5708, -1.5708]
- sample idx [0, 1] — device [-6.917529027641082e+18, -9.223372036854776e+18]

## Reproduce
```
python findings_v2/vulkan_asus_a12201/bugs/operators/repro_asin_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
