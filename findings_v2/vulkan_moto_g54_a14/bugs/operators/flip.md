# Vulkan operator bug — `flip`

**Failure mode:** mismatch  ·  **Root cause:** operator — **NON-DETERMINISTIC kernel**
**Divergence type:** wrong value  ·  **Device:** Moto G54 5G (Mali-G57), Android 14 — device-verified
**Stability:** NON-DETERMINISTIC — wrong output on 1/7 reruns, correct on 6/7 of the SAME baked input (race / uninitialized memory)

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w100:771` `out[1]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the Moto. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=2.354e+00 (rtol=0.01 atol=0.001)`
- **worst element** (idx 8): eager **-1.03194** vs device **1.32234**
- input dtypes: ['torch.float32']
- sample idx [0, 1, 2, 3, 4, 8] — eager  [1.32234, 1.32234, 1.32234, 1.32234, 0.43771, -1.03194]
- sample idx [0, 1, 2, 3, 4, 8] — device [1.32234, 1.32234, 1.32234, 1.32234, 1.32234, 1.32234]

## Reproduce
```
python findings_v2/vulkan_moto_g54_a14/bugs/operators/repro_flip.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
