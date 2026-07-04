# Ruled-out suspects (arm64 phone run)

- **`bitwise_left_shift.*`** (~715), **`pow.*`**, **`sum.IntList`** — INT64 `2^63` reference sentinels
  + shift-UB (filter 3d). Same confounds as x86; not device bugs.
- **portable-fallback mismatches** (`ops=0`: `topk`, `remainder`, `grid_sampler_2d`, `native_group_norm`,
  …) — ran on the phone's portable kernels, not the XNNPACK delegate; out of scope for the delegate.
## Resolved (not dismissed): the 2,285 `executor unavailable` "crashes"
These were **dropped-worker artifacts** — the phone worker dropping under high concurrency
(window=48), not code crashes. They were **not hand-waved away**: the full CRASH set (2,345 jobs) was
**re-run patiently** with a gentler config (window=6, timeout=120s), which resolved **all of them in a
single pass with 0 left unavailable**. Real verdicts: **2,133 OK, 82 SKIP, 70 MISMATCH, 60 genuine
native aborts**. The 70 mismatches + 82 skips were **merged into the findings** (same
`log`/`logit`/`gelu` + portable-fallback families — no new operator surfaced). Lesson: on a flaky
device, cut in-flight concurrency rather than accept dropouts. See [../skips.md](../skips.md).
