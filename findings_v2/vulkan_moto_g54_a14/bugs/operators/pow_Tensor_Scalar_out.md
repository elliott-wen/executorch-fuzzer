# Vulkan operator bug — `pow.Tensor_Scalar_out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** Moto G54 5G (Mali-G57), Android 14 — device-verified
**Stability:** GENUINE on 3/3 reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w103:733` `out[2]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the Moto. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=1.809e-01 (rtol=0.01 atol=0.001)`
- **worst element** (idx 8): eager **0.38851** vs device **0.20759**
- input dtypes: ['torch.float32', 'torch.float32']
- sample idx [0, 1, 2, 3, 4, 8] — eager  [0.20759, 0.20759, 0.20759, 0.20759, 0.20759, 0.38851]
- sample idx [0, 1, 2, 3, 4, 8] — device [0.20759, 0.20759, 0.20759, 0.20759, 0.20759, 0.20759]

## Reproduce
```
python findings_v2/vulkan_moto_g54_a14/bugs/operators/repro_pow_Tensor_Scalar_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
