# Vulkan `index_put` — writes land in the wrong positions

- **Mode:** MISMATCH · **Root cause:** operator kernel (delegated) · **Occurrences:** 213 (194 non-finite-tagged, 19 value)
- **Repro:** [repro_vulkan-index_put-mismatch.py](repro_vulkan-index_put-mismatch.py) (`w0:22`)

```
eager  : [inf, 0.513757]
device : [0.513757, inf]
```

`index_put` places the written values at the wrong indices (positions swapped here) — the Vulkan
scatter indexing is wrong. Matches the known Vulkan index_put / scatter drops-writes family.
