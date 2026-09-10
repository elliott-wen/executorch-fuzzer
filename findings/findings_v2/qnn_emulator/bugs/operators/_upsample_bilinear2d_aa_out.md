# Vulkan operator bug — `_upsample_bilinear2d_aa.out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** QNN HTP x86 emulator (fp16) — device-verified
**Stability:** GENUINE on 3/3 reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w14:86` `out[2]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=2.000e+00 (rtol=0.0 atol=0.0)`
- input dtypes: ['torch.uint8']
- eager  (CPU): [4.0, 4.0, 4.0, 3.0, 4.0]
- device (GPU): [2.0, 3.0, 3.0, 3.0, 3.0]

## Reproduce
```
python findings_v2/qnn_emulator/bugs/operators/repro__upsample_bilinear2d_aa_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
