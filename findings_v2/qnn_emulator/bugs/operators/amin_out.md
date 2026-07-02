# Vulkan operator bug — `amin.out`

**Failure mode:** mismatch  ·  **Root cause:** operator — **NON-DETERMINISTIC kernel**
**Divergence type:** wrong value  ·  **Device:** QNN HTP x86 emulator (fp16) — device-verified
**Stability:** NON-DETERMINISTIC — wrong on 3/7, correct on 4/7 of the SAME input

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w0:334` `out[3]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] shape () vs (2, 3)`
- **worst element** (idx None): eager **None** vs device **None**
- input dtypes: ['torch.float32']
- sample idx [0] — eager  [3.0]
- sample idx [0] — device [131008.00781]

## Reproduce
```
python findings_v2/qnn_emulator/bugs/operators/repro_amin_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
