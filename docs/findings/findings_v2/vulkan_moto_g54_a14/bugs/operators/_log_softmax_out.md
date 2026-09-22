# Vulkan operator bug — `_log_softmax.out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** Moto G54 5G (Mali-G57), Android 14 — device-verified
**Stability:** GENUINE on 3/3 reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w113:359` `out[3]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the Moto. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] shape (4, 4, 2) vs (4, 3, 4)`
- input dtypes: ['torch.float32']
- sample idx [0, 1, 2, 3, 4] — eager  [-1.45489, -1.32941, -1.32941, -1.26909, -1.51909]
- sample idx [0, 1, 2, 3, 4] — device [1.27289, 0.97996, -0.84106, -1.36849, -0.9192]

## Reproduce
```
python findings_v2/vulkan_moto_g54_a14/bugs/operators/repro__log_softmax_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
