# Vulkan operator bug — `pow.Tensor_Tensor_out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** QNN HTP x86 emulator (fp16) — device-verified
**Stability:** GENUINE on 3/3 reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w104:564` `out[0]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=1.000e+00 (rtol=0.01 atol=0.001)`
- **worst element** (idx 0): eager **1.0** vs device **0.0**
- input dtypes: ['torch.float32', 'torch.float32']
- sample idx [0, 1] — eager  [1.0, 1.0]
- sample idx [0, 1] — device [0.0, 0.0]

## Reproduce
```
python findings_v2/qnn_emulator/bugs/operators/repro_pow_Tensor_Tensor_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
