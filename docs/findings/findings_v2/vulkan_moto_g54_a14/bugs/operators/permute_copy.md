# Vulkan operator bug — `permute_copy`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** Moto G54 5G (Mali-G57), Android 14 — device-verified
**Stability:** GENUINE on 3/3 reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w101:871` `out[0]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the Moto. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=1.413e+00 (rtol=0.01 atol=0.001)`
- **worst element** (idx 11): eager **0.53271** vs device **-0.88037**
- input dtypes: ['torch.float16']
- sample idx [0, 1, 2, 3, 4, 11] — eager  [0.38257, -0.14319, -0.88037, -0.18811, -0.74072, 0.53271]
- sample idx [0, 1, 2, 3, 4, 11] — device [0.38257, 0.38257, -0.88037, -0.88037, -0.74072, -0.88037]

## Reproduce
```
python findings_v2/vulkan_moto_g54_a14/bugs/operators/repro_permute_copy.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
