# Vulkan operator bug — `fill.Scalar`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** ASUS a12201 (Vulkan) — device-verified
**Stability:** GENUINE on 3/3 reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w102:429` `out[1]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=1.000e+00 (rtol=0.0 atol=0.0)`
- **worst element** (idx 0): eager **1.0** vs device **0.0**
- input dtypes: ['torch.bool']
- sample idx [0, 1] — eager  [1.0, 1.0]
- sample idx [0, 1] — device [0.0, 0.0]

## Reproduce
```
python findings_v2/vulkan_asus_a12201/bugs/operators/repro_fill_Scalar.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
