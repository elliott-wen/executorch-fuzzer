# Vulkan operator bug — `index_select.out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** QNN HTP x86 emulator (fp16) — device-verified
**Stability:** GENUINE on 3/3 reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w22:615` `out[0]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] shape (2, 2, 4) vs (2, 1, 4, 2)`
- **worst element** (idx 2): eager **-68.0** vs device **1.0**
- input dtypes: ['torch.float32', 'torch.int32']
- sample idx [0, 1, 2, 3, 4] — eager  [-1.0, -4.0, -68.0, 1.0, -1.0]
- sample idx [0, 1, 2, 3, 4] — device [1.0, 1.0, 1.0, 1.0, 0.0]

## Reproduce
```
python findings_v2/qnn_emulator/bugs/operators/repro_index_select_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
