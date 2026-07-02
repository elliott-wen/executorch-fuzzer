# RULED OUT — `maximum.out`

**Status:** not filed. single-op isolation won't lower/run (7/7 device-fail); bisected OPERATOR but unconfirmable as one op.

---

# Vulkan operator bug — `maximum.out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** QNN HTP x86 emulator (fp16) — device-verified
**Stability:** GENUINE on 0/3 reruns  ⚠️ FLAKY / near-tolerance — treat as low-confidence

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w102:158` `out[1]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `build=SKIP`
- input dtypes: []
- eager  (CPU): None
- device (GPU): None

## Reproduce
```
python findings_v2/qnn_emulator/bugs/operators/repro_maximum_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
