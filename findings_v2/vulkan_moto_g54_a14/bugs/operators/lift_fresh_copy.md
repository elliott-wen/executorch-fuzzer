# Vulkan operator bug — `lift_fresh_copy`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** Moto G54 5G (Mali-G57), Android 14 — device-verified
**Stability:** HIGH — GENUINE on 7/7 re-verification reruns (earlier 3/3 flake was transient device noise)

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w100:65` `out[0]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the Moto. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=4.576e+00 (rtol=0.01 atol=0.001)`
- **worst element** (idx 24): eager **3.24801** vs device **-1.3279**
- input dtypes: ['torch.float32']
- sample idx [0, 1, 2, 3, 4, 24] — eager  [1.17719, 0.90628, -0.77782, -1.26559, 0.16879, 3.24801]
- sample idx [0, 1, 2, 3, 4, 24] — device [1.17719, 0.90628, -0.77782, -1.26559, 0.16879, -1.3279]

## Reproduce
```
python findings_v2/vulkan_moto_g54_a14/bugs/operators/repro_lift_fresh_copy.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
