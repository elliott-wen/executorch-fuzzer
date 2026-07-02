# RULED OUT — `bitwise_right_shift.Tensor_Scalar_out`

**Status:** not filed as an operator bug. reproduces eager when it runs (6/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient.

---

# Vulkan operator bug — `bitwise_right_shift.Tensor_Scalar_out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** Moto G54 5G (Mali-G57), Android 14 — device-verified
**Stability:** GENUINE on 0/3 reruns  ⚠️ FLAKY / near-tolerance — treat as low-confidence

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w109:559` `out[1]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the Moto. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `op reproduces eager alone`
- input dtypes: []
- eager  (CPU): None
- device (GPU): None

## Reproduce
```
python findings_v2/vulkan_moto_g54_a14/bugs/operators/repro_bitwise_right_shift_Tensor_Scalar_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
