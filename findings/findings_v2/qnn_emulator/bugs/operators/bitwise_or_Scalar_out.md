# Vulkan operator bug — `bitwise_or.Scalar_out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** QNN HTP x86 emulator (fp16) — device-verified
**Stability:** HIGH — GENUINE on 5/7 re-verification reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w0:498` `out[1]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] shape () vs (2, 3)`
- **worst element** (idx None): eager **None** vs device **None**
- input dtypes: ['torch.uint8']
- sample idx [0] — eager  [249.0]
- sample idx [0] — device [131008.00781]

## Reproduce
```
python findings_v2/qnn_emulator/bugs/operators/repro_bitwise_or_Scalar_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
