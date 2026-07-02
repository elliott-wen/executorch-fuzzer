# Vulkan operator bug — `gelu.out`

**Failure mode:** mismatch  ·  **Root cause:** operator — **NON-DETERMINISTIC kernel**
**Divergence type:** wrong value  ·  **Device:** QNN HTP x86 emulator (fp16) — device-verified
**Stability:** NON-DETERMINISTIC — wrong on 1/7, correct on 6/7 of the SAME input

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w103:94` `out[1]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=3.175e+04 (rtol=0.01 atol=0.001)`
- **worst element** (idx 15): eager **162754.79688** vs device **131008.00781**
- input dtypes: ['torch.float32']
- sample idx [0, 1, 2, 3, 4, 15] — eager  [1e-05, 0.00046, 6e-05, 7.38906, 148.41316, 162754.79688]
- sample idx [0, 1, 2, 3, 4, 15] — device [1e-05, 0.00046, 6e-05, 7.39844, 148.50002, 131008.00781]

## Reproduce
```
python findings_v2/qnn_emulator/bugs/operators/repro_gelu_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
