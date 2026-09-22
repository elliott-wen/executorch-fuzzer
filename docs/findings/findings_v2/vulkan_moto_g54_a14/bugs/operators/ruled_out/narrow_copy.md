# RULED OUT — `narrow_copy`

**Status:** not filed as an operator bug. reproduces eager when it runs (7/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient.

---

# Vulkan operator bug — `narrow_copy`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** shape divergence  ·  **Device:** Moto G54 5G (Mali-G57), Android 14 — device-verified
**Stability:** GENUINE on 1/3 reruns  ⚠️ FLAKY / near-tolerance — treat as low-confidence

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w104:516` `out[3]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the Moto. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] shape (2, 4, 4) vs (3, 2, 3, 2)`
- input dtypes: ['torch.float32']
- eager  (CPU): [-1.34119, -0.29624, 1.2547, 0.90448, 0.00448, -0.09844]
- device (GPU): [0.0, 1.0, 0.0, 2.0, 4.0, 1.0]

## Reproduce
```
python findings_v2/vulkan_moto_g54_a14/bugs/operators/repro_narrow_copy.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
