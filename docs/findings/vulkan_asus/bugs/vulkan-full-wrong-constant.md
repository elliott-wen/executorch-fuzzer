# Vulkan `full` — returns the wrong constant (`full(1.0)` → `0.0`)

- **Mode:** MISMATCH · **Root cause:** operator kernel / memory (delegated) · **Occurrences:** 25 (`full.out`) + 17 (`fill.Scalar`)
- **Repro:** [repro_vulkan-full-mismatch.py](repro_vulkan-full-mismatch.py) (`w110:446`)

```
eager  : [1.0]
device : [0.0]
```

`full`/`fill` must output the requested constant everywhere; Vulkan returns `0.0` instead — the
constant-fill buffer was overwritten / aliased with a live tensor. Memory-planning bug (a constant op
whose output is data-independent, so this is unambiguous).
