# Bug: reduction written into a narrower bool out= tensor diverges

## One-line statement
A reduction (`prod.int_out`, and by extension comparison/reduction ops) whose
`out=` tensor is a **narrower `bool` dtype** is handled differently by ExecuTorch
portable vs eager PyTorch: the two backends disagree on how the wide reduction
result is narrowed into `bool` (product-then-cast / saturation vs an
any-style truncation), producing flipped `True`/`False` values — e.g.
`prod.int_out(n2, 0, …, out bool)` yields `portable=False` where `eager=True` at
several positions. Unlike the int64-overflow case (UB on both sides), narrowing a
reduction into a `bool` `out=` has a defined target dtype, so this is **likely a
real ExecuTorch narrowing-out-dtype bug**.

## PyTorch / narrowing semantics
When a reduction is computed and then stored into an `out=` tensor of a narrower
dtype (here `bool`), PyTorch computes the reduction in its natural accumulate type
and then **casts each result element to the out dtype**. For `bool`, the cast is
"non-zero → `True`, zero → `False`". So `prod` over a column produces an integer
product per output position, and only a product that is exactly `0` should map to
`False`; any non-zero product maps to `True`. The result is well-defined by the
declared `out=` dtype — the two backends are expected to agree.

## Root cause
Portable and eager differ in **where** the narrowing happens relative to the
reduction:
- **Eager** computes the integer reduction (`prod`) and casts the final per-position
  product to `bool` (`!= 0`), so a non-zero product becomes `True`.
- **Portable** appears to fold the `bool` out dtype into the reduction itself —
  saturating intermediates to `{0,1}` or treating the accumulation as an `any`-style
  / boolean operation — so a column whose true integer product is non-zero can still
  land on `False`.

The net effect is a single-bit flip (`bool` delta = 1) at the positions where the
two narrowing strategies disagree. This is mechanism **M5** in
[mismatch-delta-exact.md](../mismatch-delta-exact.md): "prod/comparison written into
a bool `out` tensor … reducing into a narrower (bool) out dtype is handled
differently (any-vs-product / saturation) by the two backends."

## Reproduction
Real corpus job with a `prod` reduction into a `bool` out:
- `corpus/portable/w0/w0_101.py:39` —
  `prod.int_out(n2, 0, False, dtype=None, out=torch.empty((4, 4), dtype=torch.bool))`,
  where `n2` is an int/float-derived tensor. The narrowed `bool` result flips at
  several positions.

## Replay
Self-contained in-process replay:
[`replay_reduction-into-bool-out.py`](replay_reduction-into-bool-out.py). Loads the
stored corpus job `corpus/portable/w0/w0_101.py`, runs it on the portable
ExecuTorch runtime and against the eager reference, and asserts a `bool` divergence
(`max|delta| ≥ 1`) at USER_OUTPUT index 3 (exits 0 iff reproduced).

```
$ cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_reduction-into-bool-out.py 2>&1 \
    | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
job w0:101 out[3] dt p=torch.bool/e=torch.bool max|delta|=1.0000e+00 at idx 2
  portable=False  eager=True
REPLAY: bug REPRODUCED
```

At USER_OUTPUT index 3 the narrowed `bool` reduction reads `portable=False` where
eager reads `True` (single-bit flip, `delta = 1`) — confirming the two backends
narrow the reduction into `bool` differently.

## Scope
Per the M5 row of [mismatch-delta-exact.md](../mismatch-delta-exact.md), reductions
narrowed into a `bool` out account for **roughly dozens** of small-delta rows in the
exact-compare MISMATCH cluster (delta = 1, single-bit flips). It is a small but
distinct family, separate from the M6 comparison/bool flips that are merely
inherited from upstream float-kernel differences.

## Classification & severity
- **Class:** **Likely real ExecuTorch narrowing-out-dtype bug.** The `out=` dtype is
  declared and defined (`bool`), so there is a correct narrowing behavior
  (reduce-then-cast `!= 0`); portable's result disagrees with it.
- **Op:** `aten::prod.int_out` (and, by the same mechanism, other reductions /
  comparisons) written into a narrower `bool` `out=`.
- **Trigger:** a reduction whose `out=` tensor dtype is narrower than the reduction's
  natural accumulate type — specifically a `bool` out.
- **Severity:** **Medium.** Silent single-bit divergence (no crash); defined target
  dtype; smaller blast radius than the M3/M4 integer-semantics bugs but a genuine
  correctness gap.

## Fix
Audit narrowing-out-dtype reductions in the portable kernels: ensure the reduction
is computed in its natural accumulate type and **then** cast per-element to the
`out=` dtype (for `bool`, `result != 0`), rather than folding the narrow out dtype
into the accumulation (which can saturate / behave like `any`). Add tests that
reduce into a `bool` (and other narrowing) `out=` and compare against eager.

Cross-reference: mechanism **M5** in
[mismatch-delta-exact.md](../mismatch-delta-exact.md).
