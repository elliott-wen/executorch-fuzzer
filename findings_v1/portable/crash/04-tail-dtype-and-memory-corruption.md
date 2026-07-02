# Crash reason: long tail — dtype-fatal Checks, unhandled complex, and memory corruption

This long-tail file has been **split into per-bug files** (each now ships a self-contained
`replay_*.py`). This page is kept as an index so existing links don't break.

- **A. scalar_type mismatch fatal Check** (`tensor_util.h`, SIGABRT, ~13) →
  [05-scalar-type-mismatch.md](05-scalar-type-mismatch.md)
- **B. reduce_util.cpp path** (~10) → [06-reduce-util-unhandled-dtype.md](06-reduce-util-unhandled-dtype.md).
  VERIFIED true signal is **SIGFPE (136)**, not SIGABRT — the `reduce_util.cpp:380` log line
  is non-fatal; the job dies on an integer divide-by-zero.
- **C. unhandled ComplexFloat into generic elementwise** (`dtype_util.h`, SIGSEGV 139, 1–2) →
  [07-complexfloat-elementwise.md](07-complexfloat-elementwise.md)
- **D. memory corruption** — OOB writes / heap corruption, the most severe (SIGABRT/SIGSEGV,
  ≈5) → [08-memory-corruption.md](08-memory-corruption.md)
