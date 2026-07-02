# RULED OUT — `any.dims_out`

**Status:** not filed. reproduces eager when it runs (5/7 clean) — NOT a kernel bug.

---

# Vulkan operator bug — `any.dims_out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** dtype divergence  ·  **Device:** QNN HTP x86 emulator (fp16) — device-verified
**Stability:** GENUINE on 1/3 reruns  ⚠️ FLAKY / near-tolerance — treat as low-confidence

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w0:721` `out[0]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the device. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] dtype torch.uint8 vs torch.float32`
- input dtypes: ['torch.int64']
- eager  (CPU): [0.0]
- device (GPU): [3.0]

## Reproduce
```
python findings_v2/qnn_emulator/bugs/operators/repro_any_dims_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
