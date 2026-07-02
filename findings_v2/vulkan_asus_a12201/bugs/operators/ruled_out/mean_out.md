# RULED OUT — `mean.out`

**Status:** not filed. reproduces eager when it runs (7/7 clean) — NOT a kernel bug.

---

# Vulkan operator bug — `mean.out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** ASUS a12201 (Vulkan) — device-verified
**Stability:** GENUINE on 0/3 reruns  ⚠️ FLAKY / near-tolerance — treat as low-confidence

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w100:464` `out[2]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `op reproduces eager alone`
- input dtypes: []
- eager  (CPU): None
- device (GPU): None

## Reproduce
```
python findings_v2/vulkan_asus_a12201/bugs/operators/repro_mean_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
