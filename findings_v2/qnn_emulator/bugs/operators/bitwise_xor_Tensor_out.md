# Vulkan operator bug — `bitwise_xor.Tensor_out`

**Failure mode:** mismatch  ·  **Root cause:** operator — **NON-DETERMINISTIC kernel**
**Divergence type:** wrong value  ·  **Device:** QNN HTP x86 emulator (fp16) — device-verified
**Stability:** NON-DETERMINISTIC — wrong on 2/7, correct on 4/7 of the SAME input

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w106:238` `out[1]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `op reproduces eager alone`
- input dtypes: []
- eager  (CPU): None
- device (GPU): None

## Reproduce
```
python findings_v2/qnn_emulator/bugs/operators/repro_bitwise_xor_Tensor_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
