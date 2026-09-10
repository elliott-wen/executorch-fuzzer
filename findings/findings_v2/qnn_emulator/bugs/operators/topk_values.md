# Vulkan operator bug — `topk.values`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** QNN HTP x86 emulator (fp16) — device-verified
**Stability:** GENUINE on 3/3 reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w112:127` `out[2]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=1.417e+00 (rtol=0.01 atol=0.001)`
- **worst element** (idx 0): eager **0.93408** vs device **-0.48242**
- input dtypes: ['torch.float16', 'torch.int64']
- sample idx [0, 1, 2, 3] — eager  [0.93408, -0.48242, 1.11035, 1.16406]
- sample idx [0, 1, 2, 3] — device [-0.48242, 0.93408, 1.11035, 1.16406]

## Reproduce
```
python findings_v2/qnn_emulator/bugs/operators/repro_topk_values.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
