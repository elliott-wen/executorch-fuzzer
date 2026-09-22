# Vulkan `copy` — returns an unrelated buffer (copy-elision / aliasing)

- **Mode:** MISMATCH · **Root cause:** operator kernel / memory (delegated) · **Occurrences:** 116 (66 non-finite, 50 value)
- **Repro:** [repro_vulkan-copy-mismatch.py](repro_vulkan-copy-mismatch.py) (`w0:264`)

```
eager  : [1.163586, 0.60283, -0.806139]
device : [-0.482491, 1.110353, 0.934016]
```

`copy` should return its input verbatim, but the Vulkan output is a **completely different buffer** —
the copy was elided/aliased onto the wrong tensor. A serious memory-planning bug (matches the known
Vulkan copy-elision aliasing family).
