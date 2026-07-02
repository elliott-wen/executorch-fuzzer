# Vulkan operator bug — `clamp.Tensor_out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** ASUS a12201 (Vulkan) — device-verified
**Stability:** GENUINE on 3/3 reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w0:379` `out[0]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=1.200e+01 (rtol=0.0 atol=0.0)`
- **worst element** (idx 14): eager **12.0** vs device **0.0**
- input dtypes: ['torch.int32']
- sample idx [0, 1, 2, 3, 4, 14] — eager  [6.0, 6.0, 6.0, 6.0, 6.0, 12.0]
- sample idx [0, 1, 2, 3, 4, 14] — device [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

## Reproduce
```
python findings_v2/vulkan_asus_a12201/bugs/operators/repro_clamp_Tensor_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
