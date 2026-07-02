# Vulkan operator bug — `bitwise_left_shift.Tensor_Scalar_out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** dtype divergence  ·  **Device:** Moto G54 5G (Mali-G57), Android 14 — device-verified
**Stability:** HIGH — GENUINE on 5/7 re-verification reruns (earlier 3/3 flake was transient device noise)

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w100:163` `out[1]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the Moto. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] shape (1, 3, 2) vs (4, 4, 2)`
- input dtypes: ['torch.int32']
- sample idx [0, 1, 2, 3, 4] — eager  [0.0, 0.0, 0.0, 0.0, 0.0]
- sample idx [0, 1, 2, 3, 4] — device [-1.38629, -1.38629, -1.38629, -1.38629, -1.38629]

## Reproduce
```
python findings_v2/vulkan_moto_g54_a14/bugs/operators/repro_bitwise_left_shift_Tensor_Scalar_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
