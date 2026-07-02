# Confirmed Vulkan bugs — Moto G54 5G, Android 14

All repros are self-contained: they build the `.pte`, run it on the connected Vulkan worker
(broker `127.0.0.1:15554`), and print eager vs device. **Each was re-run green (reproduces) on this
Android-14 device** during this workflow. Repros are shared with the prior Moto run — the
cross-OS differential (`../CROSS_DEVICE_DIFF.md`) shows the bug surface is OS-version-independent.

| bug | failure mode | root cause | repro |
|-----|--------------|------------|-------|
| scatter.src_out drops writes | mismatch | operator | `repro_vulkan-scatter-src-drops-writes.py` |
| index_put drops writes | mismatch | operator | `repro_vulkan-index-put-drops-writes.py` |
| floor_divide(x,x) wrong | mismatch | operator | `repro_vulkan-floor-divide-self.py` |
| bitwise shift over-width | mismatch | operator | `repro_vulkan-left-shift-overwidth.py` |
| copy-elision / aliasing | mismatch + crash | graph-optimization | `repro_vulkan-copy-elision-aliasing.py` |

See each bug's `.md` for detail. Skip-class coverage gaps (layer_norm / group_norm / batch_norm /
multi-dim reductions / binary-broadcast / missing dtype shaders / extract_scalar) are enumerated in
`../WORKFLOW_RESULTS.md` and confirmed by single-op isolation in `../ISOLATION_PROBES.md`.

**NOT filed** (investigated, correct in isolation → ruled out as operator bugs): `unfold_copy`,
`narrow_copy`, `clone`, `alias_copy`, `lift_fresh_copy`, `transpose_copy`, `expand_copy` — these
crash only inside larger graphs and are the copy-elision aliasing graph-optimization confound.
