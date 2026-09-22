# Ruled out — portable-fallback crashes (not ENN)

All 85 CRASHes ran on the **portable CPU kernel** (`ops=0`), not the ENN delegate. Filter 3a.
Native aborts (no catchable message). These are portable-kernel bugs, out of scope for the
Samsung/ENN backend, but worth a portable-backend follow-up (a kernel should reject bad input
with a catchable error, not `abort()`):

| count | operator | note |
|--:|---|---|
| 78 | `max_pool2d_with_indices_backward.grad_input` | backward/training op; ENN never delegates it |
| 3 | `unfold_copy` | |
| 3 | `narrow_copy.out` | |
| 1 | `narrow_copy` | |
