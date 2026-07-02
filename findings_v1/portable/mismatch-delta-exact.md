# MISMATCH cluster: exact-compare value deltas (rtol=0), ~3,202

> **Per-bug splits (each with a runnable replay):** the mechanisms below now have dedicated
> write-ups — M1 shift → [bugs/bitwise-shift-negative.md](bugs/bitwise-shift-negative.md);
> M2 prod/pow overflow → [bugs/prod-pow-int64-overflow.md](bugs/prod-pow-int64-overflow.md);
> M3 remainder → [bugs/remainder-sign.md](bugs/remainder-sign.md);
> M4 sum → [bugs/sum-cast-order.md](bugs/sum-cast-order.md);
> M5 reduction-into-bool → [bugs/reduction-into-bool-out.md](bugs/reduction-into-bool-out.md);
> M6 comparison flips → [bugs/comparison-bool-flips.md](bugs/comparison-bool-flips.md).
> Full index: [REPLAY.md](REPLAY.md).

## Summary

This cluster is **3,214 outputs** (filtered `MISMATCH` rows with `max|delta|=… (rtol=0.0 atol=0.0)`)
from `tmp/run_portable/skip_reasons_portable.tsv`. These are integer / bool / exactly-compared
tensors whose **values** diverge between ExecuTorch portable kernels and eager PyTorch.

Split by magnitude:
- **huge (|delta| ≥ 1e15): 1,336** — ≈2^61–2^63 scale → signed-int64 wrap / undefined shift.
- **small (1 ≤ |delta| < 1e15): 1,878** — off-by-small integers, modulo sign flips, single-bit
  boolean flips.

Every divergence traces to **integer / bitwise / rounding semantics** in a handful of arithmetic
kernels, or is **inherited** by a downstream structural op (view / copy / scatter / reduction) from
one of those arithmetic kernels. Each mismatched `out[i]` was mapped to the i-th *starred* (`*=`,
returned) node in the op-chain; the harness `tmp/run_portable/cmp_delta.py` (adapted from
`cmp_nonfinite.py`) was used to print the actual portable-vs-eager values at the max-delta element
and to confirm the triggering operands.

## Dominant ops

Tally over all 3,214 rows, attributing each row to the op of its *starred returned* node
(`out[i]` → i-th `nN*=op`). Script: scratchpad `tally.py`.

| op (starred output)            | total | huge ≥1e15 | small |
|--------------------------------|------:|-----------:|------:|
| bitwise_left_shift             |  614  |   578      |  36   |
| sum                            |  436  |   151      | 285   |
| remainder                      |  385  |     1      | 384   |
| prod                           |  360  |   191      | 169   |
| bitwise_right_shift            |   78  |    15      |  63   |
| logical_xor / lt / gt / and / ge / le / eq / ne / logical_not (bool) | ~380 | ~1 | ~379 |
| bitwise_xor / bitwise_or / bitwise_and | ~80 | ~31 | ~49 |
| pow                            |   25  |    23      |   2   |
| trunc / ceil / round           |   60  |    35      |  25   |
| structural (slice_scatter, squeeze_copy, unsqueeze_copy, detach_copy, flip, alias_copy, t_copy, copy, roll, pixel_unshuffle, scatter, select_scatter, max, min, amax, any, abs, sub, add, linear, sign …) | ~1,035 | ~430 | ~605 |

Mechanism rollup (each row attributed to one root cause; structural/copy ops are folded into the
upstream arithmetic op they inherit from — see §Inherited):

| mechanism bucket                              | rows |
|-----------------------------------------------|-----:|
| shift (negative count UB / count-mask)        |  692 |
| sum dtype/accumulate-then-cast vs cast-then-accumulate | 436 |
| integer div/mod sign + rounding (remainder/fmod/floor_divide/div) | 400 |
| comparison/bool flips (inherited from upstream float diff) | 353 |
| int64 overflow (prod / pow)                   |  214 |
| rounding into int                             |   53 |
| bitwise on already-overflowed/shifted operand |   31 |
| structural ops inheriting an upstream arith divergence ("other") | ~1,035 |

Derivation note: 692/3214 directly-starred shift outputs; among 1,546 jobs that have a shift on
*any* output, **881 contain a literal negative shift constant** in the source (`<<−2`, `<<−3`, …),
the rest use a tensor shift count that can be negative at runtime (`n0 << n0` with `n0<0`).

## Mechanisms (reproduced)

### M1 — Negative / out-of-range shift count (left & right shift). DOMINANT huge cause.
C++ `<<`/`>>` by a negative or ≥64 count is **undefined behavior**. Eager's CPU kernel *clamps the
result to 0*; the portable kernel evaluates `x << (count & 63)` (x86 masks the shift amount), giving
a huge bogus value.

- `w0:1066` — `corpus/portable/w0/w0_1066.py:33`
  `n4 = bitwise_left_shift.Tensor_Scalar(n1, -3)`, n1 a {0,1} bool→int.
  `out[0]`: **portable=2305843009213693952 (=1<<61)  eager=0**  (all elements).
- `w0:1388:32` `bitwise_left_shift(n0, -2)` → portable=4611686018427387904 (=1<<62), eager=0.
- `w0:454:44`  `bitwise_left_shift(n6, -2)` → portable=4611686018427387904, eager=0.
- `w3:2344:28` `bitwise_left_shift.Tensor_out(n0, n0)` with n0=-3 → `(-3)<<(-3)`, negative *count*
  via tensor operand → portable=-6917529027641081853, eager=3 (after the following xor).

Standalone confirmation:
```
eager  x << -3  -> [0,0,0,0,0]      (clamped to 0)
python 1 << 61  -> 2305843009213693952   == portable's value
eager  x >> -2  -> [...,-1]         (arithmetic, defined differently again)
```

### M2 — int64 overflow in prod / pow (DOMINANT huge cause #2).
`prod`/`pow` on int64 overflow 2^63; portable and eager wrap via different arithmetic, so the
wrapped bit-patterns disagree.

- `w0:1492` — `corpus/portable/w0/w0_1492.py:41` `prod.int_out(n12, 0, …, out int64)`
  `out[0][1]`: **portable=-9223372036854775808 (INT64_MIN)  eager=429025**.
- `w0:570:31`  `prod.int_out(n1,…)` → portable=-9223372036854775808, eager=0.
- `w0:1198:41` `pow.Tensor_Scalar_out(n7, 3, out int64)`, n7 a large int64
  → **portable=-9223372036854775808  eager=0** (both overflow, different wrap).

### M3 — Integer remainder sign convention (DOMINANT small cause).
`torch.remainder` follows the **divisor's sign** (Python `%`: result sign = divisor). Portable's
kernel returns the wrong sign (C `fmod`-style, sign of dividend) for negative divisors / bool inputs.

- `w0:1607` — `corpus/portable/w0/w0_1607.py:51` `remainder.Scalar(n2, -8)`, n2 bool(0/1).
  `out[2]`: **portable=1  eager=-7/-8** (delta up to 9).
- `w0:452:45` `remainder.Scalar(n13, -3)` → **portable=1  eager=-2** (delta 3).

Standalone:
```
remainder(x,-8) eager -> [0,-7,-6,-1,-1,-7]   (sign follows divisor → negative)
fmod(x,-8)      eager -> [0, 1, 2, 7,-1,-7]   (sign follows dividend; matches portable's +1)
```
i.e. portable computes `fmod` semantics where eager `remainder` is requested.

### M4 — sum reduction: cast-then-accumulate vs accumulate-then-cast (DOMINANT small cause #2).
`sum.IntList_out` with a **float input and int64 `out`** dtype. Eager casts each element to the
output integer type *first* (truncating toward zero per-element) then sums integers; portable
accumulates in float then casts the total. The truncation residuals accumulate to a different total.

- `w0:707` — `corpus/portable/w0/w0_707.py:43`
  `sum.IntList_out(n0:float16, None, True, out int64)`, n0 ∈ {0.07,5.84,6.83,…}.
  `out[5]`: **portable=116  eager=97**.
  Proof: `n0.to(int64).sum() = 97` (eager path); `int(n0.sum()) = 116` (portable path);
  `n0.double().sum() = 116.85`.

### M5 — prod/comparison written into a bool `out` tensor.
- `w0:101` — `corpus/portable/w0/w0_101.py:39`
  `prod.int_out(n2, 0, …, out bool)` → `out[3]`: **portable=False  eager=True** at several
  positions. Reducing into a narrower (bool) out dtype is handled differently
  (any-vs-product / saturation) by the two backends.

### M6 — Comparison / boolean flips inherited from an upstream float kernel diff.
`lt/gt/ge/le/eq/logical_*` outputs that flip True↔False (delta=1) because an upstream float op
(rsqrt, layernorm, batch_norm, etc.) produced a slightly different value that crosses the
comparison threshold. The comparison kernel itself is correct.

- `w0:227` — `corpus/portable/w0/w0_227.py:33-34` `n3=rsqrt(...)`, `n4=lt.Tensor_out(n3, L3,
  out int64)`. `out[1]`: **portable=1  eager=0**; the sibling float output `out[3]` (rsqrt-derived)
  differs 1.57 vs 2.36, confirming the input — not the comparator — diverged.

### Inherited (the large "other" bucket, ~1,035 rows)
Starred outputs whose op is a pure view/copy/reshape/scatter/structural reduction
(`slice_scatter, squeeze_copy, unsqueeze_copy, detach_copy, flip, alias_copy, t_copy, copy, roll,
pixel_unshuffle, scatter, select_scatter, max, min, amax, any, abs, sub, sign, …`). These have no
integer arithmetic of their own, so their value divergence is **inherited** from an upstream M1/M2/M3
node.
- `w28:577` — `corpus/portable/w28/w28_577.py:44` `n18 = slice_scatter(n15, …)` where
  `n15 = prod.int_out(...)` (line 41). The huge slice_scatter delta is the prod overflow (M2)
  carried through the scatter.

## Classification & severity

| # | Mechanism | Class | Approx share | Severity |
|---|-----------|-------|-------------:|----------|
| M1 | Negative/oversized shift count | **C++ UB**, divergence both-undefined. Generator artifact feeds negative constants/operands. | ~692 starred + many inherited (huge ≈40% of cluster) | Low–Med. Not a correctness bug per se (UB), but ET should match eager's *clamp-to-0* to be safe and deterministic. **Genuinely worth a kernel guard.** |
| M2 | prod/pow int64 overflow | C++ signed-overflow UB; both wrap, bit-patterns differ. | ~214 starred + inherited | Low. UB on both sides; flag but not a clear ET bug. |
| M3 | remainder negative-divisor sign | **Real ExecuTorch kernel bug** — portable returns `fmod` sign where `remainder` (divisor-sign) is specified. | ~385 (mostly small) | **High.** Defined PyTorch semantics; portable is wrong. |
| M4 | sum float-in/int-out accumulate-vs-cast order | **Real ExecuTorch kernel bug** — eager casts-then-accumulates per element; portable accumulates-then-casts. | ~436 | **High.** Defined, reproducible, wrong totals. |
| M5 | reduction into bool out | Likely real ET bug (narrowing-out handling). | small (~dozens) | Med. |
| M6 | comparison/bool flips | Inherited from upstream float diff; comparator correct. | ~353 | Low. Not a bug in the integer op; symptom of float kernel diffs. |
| Inherited structural | view/copy/scatter carrying M1/M2/M3 | Not new bugs. | ~1,035 | n/a (dedup to root). |

Estimated share: genuine ExecuTorch integer-semantics bugs (M3+M4+M5) ≈ **27%** of starred rows
(~880); C++ UB / generator-fed negative shifts & overflow (M1+M2) ≈ **28%** (~906) plus their
inherited copies; comparison/structural inheritance (M6 + "other") ≈ **45%**, which dedup to the
above roots.

## Recommendation

1. **Fix `remainder` (M3, high):** portable's integer/bool `remainder` must follow PyTorch
   divisor-sign semantics (`r = a - floor(a/b)*b`), not C `fmod`. Add negative-divisor and bool-input
   tests. Highest-value fix — ~385 small-delta cases.
2. **Fix `sum.IntList_out` mixed float-in / int-out (M4, high):** match eager's *cast each element to
   the out integer type, then accumulate* order (or document/forbid the dtype mismatch). ~436 cases.
3. **Guard shifts (M1):** clamp negative or ≥bitwidth shift counts to the eager result (left-shift →
   0) instead of relying on hardware `count & 63` UB; or reject at lowering. Removes ~40% of the huge
   cluster. Also have the corpus generator stop emitting negative shift constants to separate genuine
   bugs from UB noise.
4. **prod/pow overflow (M2):** lower priority (UB both sides); optionally document that int64
   reductions/powers that overflow are unspecified.
5. **Reduction into bool out (M5):** audit narrowing-out-dtype reductions.
6. **Comparison/structural (M6 + inherited):** not actionable here; they are downstream of the float
   and integer root causes above — fixing M1–M5 (plus the float-kernel clusters owned by sibling
   agents) collapses most of these.
