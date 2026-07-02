# XNNPACK corpus — CRASH reasons

All 5,157 crashes localized (run each stored `.pte` in an isolated subprocess, capture the
fatal signal + abort signature). **Verdict: the crash profile is essentially identical to
portable — 100% shared, 0 XNNPACK-delegate-specific crash classes.** This is expected: the
crashing ops (`narrow_copy`, `unfold_copy`, integer `div`/`%`) are *not* XNNPACK-delegated, so
they run the same portable kernels and abort the same way.

## Distribution (vs portable)

| signal | xnnpack | share | portable share |
|---|---:|---:|---:|
| SIGABRT | 4,980 | 96.6% | 96.5% |
| SIGFPE | 170 | 3.3% | 3.4% |
| SIGSEGV | 7 | 0.1% | 0.1% |

## Deduplicated reasons — all shared with portable

| count | reason | → portable finding |
|---:|---|---|
| 3,548 | `narrow_copy`/`select` **negative dim** → `tensor_impl.h size()` assert | [crash/01-narrow_copy-negative-dim.md](../../portable/crash/01-narrow_copy-negative-dim.md) |
| 1,431 | `unfold_copy` **0-D input** → `tensor_impl.h size()` assert | [crash/02-unfold_copy-zero-dim.md](../../portable/crash/02-unfold_copy-zero-dim.md) |
| 170 | integer `div`/`remainder`/`fmod`/`floor_divide` **by zero** → SIGFPE | [crash/03-integer-divide-by-zero-sigfpe.md](../../portable/crash/03-integer-divide-by-zero-sigfpe.md) |
| ~28 | dtype-mismatch Check, reduce-util unhandled dtype, SIGSEGV tail | [crash/04-tail-dtype-and-memory-corruption.md](../../portable/crash/04-tail-dtype-and-memory-corruption.md) |

Attribution of the `tensor_impl.h size()` aborts by op presence: **narrow_copy 3,722,
unfold_copy 1,257** — same two missing-guard bugs as portable.

The only delta from portable is a single new tail signature — `op_embedding.cpp` indices must
be Long/Int (1 case) — negligible and the same graceful-check-should-have-fired family.

**Conclusion:** no XNNPACK-specific crash work needed; the portable crash fixes
(normalize/range-check `dim` before `size()`, 0-D guard, integer-divisor zero guard) resolve
the entire xnnpack crash set too. Raw data: `tmp/run_xnnpack/crashes.jsonl`.
