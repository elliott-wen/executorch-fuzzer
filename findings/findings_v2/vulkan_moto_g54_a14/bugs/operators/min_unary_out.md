# Vulkan operator bug — `min.unary_out`

**Failure mode:** mismatch  ·  **Root cause:** operator (kernel diverges in isolation)
**Divergence type:** wrong value  ·  **Device:** Moto G54 5G (Mali-G57), Android 14 — device-verified
**Stability:** GENUINE on 3/3 reruns

## Evidence — single-op isolation with exact baked inputs
Bisected from corpus job `w102:318` `out[2]`. The op was rebuilt as a **one-op graph fed its
exact runtime inputs** (baked from the full graph's eager intermediates, per analysis.md Phase-2),
lowered to Vulkan, and run on the Moto. It diverges as the **sole** output → the kernel is wrong on
its own (not a graph-optimization confound, not nan/inf propagation — inputs were finite).

- divergence: `out[0] max|delta|=1.246e+11 (rtol=0.01 atol=0.001)`
- **worst element** (idx 0): eager **124853944320.0** vs device **299892736.0**
- input dtypes: ['torch.int64']
- sample idx [0] — eager  [124853944320.0]
- sample idx [0] — device [299892736.0]

## Reproduce
```
python findings_v2/vulkan_moto_g54_a14/bugs/operators/repro_min_unary_out.py
```
Builds the `.pte`, runs it on the connected Vulkan worker (broker `127.0.0.1:15554`), prints eager
vs device. The baked input graph is embedded in the repro (self-contained, deterministic).
