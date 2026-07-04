# Ruled-out / needs-more-verification (vulkan ASUS run)

- **`min.dim`/`max.dim` (410), `topk`, `sum.IntList`** — these carry int64 index/value outputs and are
  dominated by `2^63` INT64 sentinels + `shape` diffs (filter 3d). Treated as **likely reference
  confounds**, not device-verified as value bugs — investigate the int64 index component before filing.
- **`bitwise_left_shift.*` / `pow` int cases** — INT64 sentinel + shift-UB confounds.
- **portable-fallback mismatches** (`ops=0`, 1,100 of 3,704) — ran on the phone's portable kernels,
  not the Vulkan delegate; out of scope for the Vulkan backend.
- **~139 `executor unavailable` "crashes"** — dropped-worker artifacts (gentle window=8 kept them low);
  re-run to real verdicts, not counted as crashes.
