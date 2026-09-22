# Vulkan operator bug — `cos.out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** ASUS a12201 (Vulkan) — device-verified
**Stability:** GENUINE on 3/3 reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w100:594` `out[4]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=1.012e+00 (rtol=0.01 atol=0.001)`
- **worst element** (idx 0): eager **0.0118** vs device **-1.0**
- input dtypes: ['torch.float32']
- sample idx [0] — eager  [0.0118]
- sample idx [0] — device [-1.0]

## Reproduce
```
python findings_v2/vulkan_asus_a12201/bugs/operators/repro_cos_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
