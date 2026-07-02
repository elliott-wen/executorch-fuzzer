# Vulkan operator bug — `rsub.Scalar`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** Moto G54 5G (Mali-G57), Android 14 — device-verified
**Stability:** GENUINE on 3/3 reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w0:705` `out[0]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the Moto. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=4.000e+00 (rtol=0.01 atol=0.001)`
- **worst element** (idx 4): eager **5.0** vs device **1.0**
- input dtypes: ['torch.uint8']
- sample idx [0, 1, 2, 3, 4] — eager  [1.0, 2.0, 4.0, 1.0, 5.0]
- sample idx [0, 1, 2, 3, 4] — device [1.0, 2.0, 1.0, 2.0, 1.0]

## Reproduce
```
python findings_v2/vulkan_moto_g54_a14/bugs/operators/repro_rsub_Scalar.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
