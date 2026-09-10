# RULED OUT — `alias_copy`

**Status:** not filed as an operator bug. reproduces eager when it runs (6/7 clean) — NOT a kernel bug; the single earlier GENUINE was near-tolerance/transient.

---

# Vulkan operator bug — `alias_copy`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** dtype divergence  ·  **Device:** Moto G54 5G (Mali-G57), Android 14 — device-verified
**Stability:** GENUINE on 1/3 reruns  ⚠️ FLAKY / near-tolerance — treat as low-confidence

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w104:346` `out[0]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the Moto. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] dtype torch.float32 vs torch.bool`
- input dtypes: ['torch.float32']
- eager  (CPU): [1.0, -1.0]
- device (GPU): [0.0, 0.0]

## Reproduce
```
python findings_v2/vulkan_moto_g54_a14/bugs/operators/repro_alias_copy.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
