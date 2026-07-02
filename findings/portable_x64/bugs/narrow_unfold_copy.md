# `narrow_copy` / `unfold_copy` — native CRASH

- **Mode:** CRASH · **Occurrences:** `narrow_copy` 2, `unfold_copy` 2

Both take a native abort (executor process dies) rather than raising a catchable error. The corpus
samples that crashed carried a non-finite leaf, but the likely trigger is a degenerate shape argument
(negative/zero dim, out-of-range length) reaching the copy kernel without a bounds guard — the
portable kernel should validate and return an error, not abort. Low count; record the crashing form
when filing.
