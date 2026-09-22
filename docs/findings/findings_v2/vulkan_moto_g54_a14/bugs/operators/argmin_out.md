# Vulkan operator bug — `argmin.out`

**Failure mode:** mismatch  ·  **Root cause:** operator — **NON-DETERMINISTIC kernel**
**Divergence type:** wrong value  ·  **Device:** Moto G54 5G (Mali-G57), Android 14 — device-verified
**Stability:** NON-DETERMINISTIC — wrong output on 2/7 reruns, correct on 2/7 of the SAME baked input (race / uninitialized memory)

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w103:290` `out[1]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the Moto. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] shape () vs (2, 2, 4)`
- input dtypes: ['torch.float32']
- sample idx [0] — eager  [105.0]
- sample idx [0] — device [0.0]

## Reproduce
```
python findings_v2/vulkan_moto_g54_a14/bugs/operators/repro_argmin_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
