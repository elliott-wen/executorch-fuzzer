# Bug: prod / pow int64 overflow — portable and eager wrap differently (UB both sides)

## One-line statement
`prod` / `pow` reductions on `int64` operands that overflow `2^63` invoke
**signed-integer overflow**, which is **undefined behavior in C++ on both sides**.
ExecuTorch portable and eager PyTorch each wrap the overflowed accumulator, but via
different arithmetic, so the wrapped bit-patterns disagree — e.g.
`prod.int_out(...)` returns `-9223372036854775808` (`INT64_MIN`) on portable vs
`429025` on eager for the same input. This is **not a clear ExecuTorch correctness
bug**: it is UB on both sides; flag, do not fix.

## PyTorch / UB semantics
`torch.prod` over an `int64` tensor accumulates in `int64`; `torch.pow` on `int64`
likewise produces an `int64` result. When the true mathematical product/power
exceeds `INT64_MAX = 2^63 − 1`, the result is **not defined** by either PyTorch's
docs or the C++ standard: signed-integer overflow is undefined behavior. The
observed value depends on the compiler, the accumulation order, intermediate
widths, and vectorization — none of which the two backends are obligated to match.
The disagreement is therefore expected and unspecified, not a divergence from a
defined reference result.

## Root cause
Both kernels accumulate the product (or compute the power) in a signed `int64`
register and let it wrap on overflow:
- **Portable**: `prod.int_out` accumulates the elements in `int64`; the overflow
  wraps to `INT64_MIN`.
- **Eager**: PyTorch's CPU reduction uses a different accumulation order / chunking
  and wraps to a different residue.

Neither side traps, saturates, or promotes to a wider type, so each produces a
self-consistent but mutually disagreeing wrapped value. The root cause is the
**C++ signed-overflow UB**, exercised because the fuzzer feeds int64 reductions
whose true product exceeds `2^63`. The `pow` variant (M2 in
[mismatch-delta-exact.md](../mismatch-delta-exact.md)) is identical in nature —
e.g. `pow.Tensor_Scalar_out(n7, 3, out int64)` → portable `INT64_MIN`, eager `0`.

## Reproduction
Real corpus job with an overflowing int64 `prod`:
- `corpus/portable/w0/w0_1492.py:41` —
  `prod.int_out(n12, 0, False, dtype=None, out=torch.empty((2,), dtype=torch.int64))`,
  where `n12` is the output of a batch-norm whose values multiply past `2^63`.

The same file also carries the `pow` variant
(`pow.Tensor_Tensor_out` at line 31, `pow.Tensor_Scalar_out` style at line 48),
which overflows analogously. Other M2 examples:
`w0:570:31` (`prod.int_out` → portable `INT64_MIN`, eager `0`) and
`w0:1198:41` (`pow.Tensor_Scalar_out(n7, 3, out int64)` →
portable `INT64_MIN`, eager `0`).

## Replay
Self-contained in-process replay:
[`replay_prod-pow-int64-overflow.py`](replay_prod-pow-int64-overflow.py). Loads the
stored corpus job `corpus/portable/w0/w0_1492.py`, runs it on the portable
ExecuTorch runtime and against the eager reference, and asserts a huge
(`≥1e15`) value divergence at USER_OUTPUT index 0 (exits 0 iff reproduced).

```
$ cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_prod-pow-int64-overflow.py 2>&1 \
    | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
job w0:1492 out[0] dt p=torch.int64/e=torch.int64 max|delta|=9.2234e+18 at idx 1
  portable=-9223372036854775808  eager=429025
REPLAY: bug REPRODUCED
```

The overflowed `int64` product wraps to `INT64_MIN` (`-9223372036854775808`) on
portable while eager wraps to `429025` — a `max|delta|` of `≈9.22e18`, exactly the
`2^63` scale expected of a signed-int64 wrap.

## Scope
Per the M2 row of [mismatch-delta-exact.md](../mismatch-delta-exact.md), the
int64-overflow (`prod` / `pow`) mechanism accounts for **~214 starred rows** in the
exact-compare MISMATCH cluster (plus additional inherited rows where a structural
op — `slice_scatter`, `copy`, etc. — carries the overflowed value downstream).
These are concentrated in the "huge" band (`|delta| ≥ 1e15`), where they are the
second-largest huge cause after negative-shift UB.

## Classification & severity
- **Class:** C++ **signed-integer-overflow UB on BOTH sides**. Both portable and
  eager wrap; their bit-patterns differ because of accumulation order, not because
  either kernel is "wrong" against a defined reference.
- **Op:** `aten::prod.int_out`, `aten::pow.Tensor_Tensor_out`,
  `aten::pow.Tensor_Scalar_out` — int64 compute type only.
- **Trigger:** int64 reduction / power whose true mathematical result exceeds
  `INT64_MAX` (`2^63 − 1`).
- **Severity:** **Low.** Not a clear ExecuTorch correctness bug — the result is
  unspecified on both sides. Flag-not-fix. It is a fuzzer/generator artifact:
  well-formed exported models do not rely on overflowing int64 products.

## Fix
Optional and low priority (since both sides are UB):
- **Document / forbid** overflowing int64 reductions and powers as unspecified.
- Optionally have the corpus generator avoid emitting int64 `prod`/`pow` chains
  whose product provably overflows `2^63`, so genuine kernel bugs are not masked by
  UB noise.
- If determinism were ever desired, a kernel could detect overflow and saturate or
  raise — but matching eager's particular wrapped residue is **not** a worthwhile
  goal.

Cross-reference: mechanism **M2** in
[mismatch-delta-exact.md](../mismatch-delta-exact.md).
