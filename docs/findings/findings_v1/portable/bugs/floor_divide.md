# Bug: aten::floor_divide — by-zero and integer floor-vs-trunc divergence

## One-line statement
ExecuTorch's portable `floor_divide` functor computes `b==0` by sign of the
dividend (`signbit(a) ? -inf : +inf`) instead of the IEEE result `a/b`, so
`0.0 // 0.0` returns `+inf` (and `-0.0 // 0.0` returns `-inf`) where eager
PyTorch returns `nan`; the rest of the float formula also omits the CPython-style
`floor`+0.5 rounding fix-up that ATen applies. The integer path is correct
(true floored division), so there is **no integer floor-vs-trunc bug in this kernel**.

## PyTorch-defined semantics
Float `floor_divide` uses the IEEE quotient at the by-zero point and otherwise
floors the true quotient; integer `floor_divide` is floored division (rounds
toward −∞, not C trunc toward zero).

Standalone eager confirmation (`.venv/bin/python`):
```
a=[0.,1.,-1.,5.]  b=[0.,0.,0.,2.]  -> torch.floor_divide(a,b) = [nan, inf, -inf, 2.]
ai=[-7,7,-7,7]    bi=[2,2,-2,-2]   -> torch.floor_divide(ai,bi) = [-4, 3, 3, -4]
```
Float by-zero is `[nan, inf, -inf, ...]` (note `0.0//0.0 = nan`). Integer result
is `[-4, 3, 3, -4]` — floored, **not** truncated (trunc would give `[-3, 3, 3, -3]`).

ATen reference (the semantics the portable kernel must match):
`pytorch_ref/c10/util/generic_math.h:34-58` `div_floor_floating`:
```cpp
35: inline C10_HOST_DEVICE scalar_t div_floor_floating(scalar_t a, scalar_t b)
36:     __ubsan_ignore_float_divide_by_zero__ {
37:   if (C10_UNLIKELY(b == 0)) {
38:     // Divide by zero: return standard IEEE result
39:     return a / b;                       // 0/0 -> nan, 1/0 -> inf, -1/0 -> -inf
40:   }
42:   auto mod = std::fmod(a, b);
43:   auto div = (a - mod) / b;
44:   if ((mod != 0) && (b < 0) != (mod < 0)) { div -= scalar_t(1); }
48:   scalar_t floordiv;
49:   if (div != 0) {
50:     floordiv = std::floor(div);
51:     if (div - floordiv > scalar_t(0.5)) { floordiv += scalar_t(1.0); }  // rounding fix-up
54:   } else { floordiv = C10_COMPAT_COPYSIGN(scalar_t(0), a / b); }       // signed zero
57:   return floordiv;
```
`div_floor_integer` (generic_math.h:61) returns 0 at `b==0`, special-cases
`INT_MIN / -1`, and subtracts 1 from truncated quotient when signs differ — i.e.
floored division.

## Portable kernel root cause
`pytorch_ref/executorch/kernels/portable/cpu/util/math_util.h:44-59`,
`floor_divide` (float overload):
```cpp
48: FLOAT_T floor_divide(FLOAT_T a, FLOAT_T b) {
49:   if (b == 0) {
50:     return std::signbit(a) ? static_cast<FLOAT_T>(-INFINITY)
51:                            : static_cast<FLOAT_T>(INFINITY);   // BUG: 0.0->+inf, -0.0->-inf; never nan
52:   }
53:   const auto mod = std::fmod(a, b);
54:   auto div = (a - mod) / b;
55:   if ((mod != 0) && std::signbit(b) != std::signbit(mod)) {
56:     return div - 1;
57:   }
58:   return div;                                                  // no floor()/0.5 rounding fix-up
59: }
```
Two divergences vs ATen:
1. **By-zero (the confirmed mismatch):** line 49-51 branches on `signbit(a)`
   and *always* returns ±inf. For `a==0.0` (`signbit(0.0)==false`) it returns
   `+inf`; for `a==-0.0` it returns `-inf`. Eager/ATen return `a/b` directly, so
   `0.0/0.0 = nan`. This is exactly the `w0:1515` case (an all-zero divisor
   `n16 = clamp(...,0,0)` fed to `floor_divide`, with a zero-containing dividend).
2. **Rounding fix-up missing:** lines 54-58 return `div` (or `div-1`) directly,
   omitting ATen's `floor(div)` + `(div-floor>0.5 ? +1)` correction (generic_math.h
   50-53). This produces small fp/fp16 rounding deltas on large magnitudes.

The integer overload (math_util.h:34-42) IS correct floored division
(`quot`, minus 1 when signs differ and remainder nonzero) and matched eager in
reproduction.

The same buggy functor is reached by two ops:
- `op_floor_divide.cpp:69` — `return utils::floor_divide(val_a, val_b);`
- `op_div.cpp:157` (floor rounding mode) — `value = utils::floor_divide(val_a, val_b);`
  (export lowered `torch.floor_divide` on integers to `aten::div.out_mode`.)
Integer by-zero in both is a CHECK (`div_by_zero_error`, op_floor_divide.cpp:64-66,
op_div.cpp:146-148) that aborts the method — eager raises `ZeroDivisionError`, so
both error out (consistent behavior, surfaces as CRASH not MISMATCH).

## Reproduction
Minimal exported graph (`torch.floor_divide`) run through the portable runtime
(`mobile.executor.et_runner.run_pte`), inputs identical to eager:

FLOAT32, `a=[0,1,-1,5] b=[0,0,0,2]`:
```
  portable: [inf, inf, -inf, 2.0]
  eager:    [nan, inf, -inf, 2.0]
```
FLOAT16, same inputs:
```
  portable: [inf, inf, -inf, 2.0]
  eager:    [nan, inf, -inf, 2.0]
```
→ Position 0 (`0.0 // 0.0`): **portable `inf`, eager `nan`** — confirmed.

INTEGER, `a=[-7,7,-7,7] b=[2,2,-2,-2]` (lowered to `div.out_mode`):
```
  portable: [-4, 3, 3, -4]
  eager:    [-4, 3, 3, -4]   (floored; trunc would be [-3,3,3,-3])
```
→ Integer floored division **matches** — no floor-vs-trunc bug here.

INTEGER by-zero, `a=[5,-5] b=[0,0]`: portable aborts the method
(`Check failed (!div_by_zero_error): ... integer division by zero`,
op_div.cpp:169); eager raises `ZeroDivisionError`. Both error — consistent.

C++ functor-level confirmation (portable math_util.h vs ATen generic_math.h,
compiled and run directly):
```
a= 0.0 b= 0.0  port=   inf  aten=  -nan
a= 1.0 b= 0.0  port=   inf  aten=   inf
a=-1.0 b= 0.0  port=  -inf  aten=  -inf
a= 5.0 b= 2.0  port=   2.0  aten=   2.0
-0.0//0.0:     port=-inf    aten=-nan
```

The corpus job `corpus/portable/w0/w0_1515.job` (`n17*=floor_divide.out(n9,n16,·)`,
`n16` all-zero divisor) did not re-trigger the nan/inf delta on replay (the
sampled dividend at the zero divisor was nonzero, and the prompt notes out=
broadcast-shape replay issues), so the divergence is demonstrated via the
deterministic minimal repro above, which is byte-identical in kernel path.

## Scope
From `tmp/run_portable/skip_reasons_portable.tsv`:
- `MISMATCH` rows whose graph contains `floor_divide`: **2468**
  (`awk -F'\t' '$1=="MISMATCH" && $4 ~ /floor_divide/' | wc -l`).
- `CRASH` rows containing `floor_divide`: **1046** (native abort — includes the
  integer/zero-divisor CHECK aborts; consistent-error, not a value divergence).

Mechanism split of the 2468 MISMATCH rows (by reason; floor_divide is one op in
a chain, so these are upper bounds on attribution):
| reason category | count | attribution |
|---|---|---|
| non-finite (nan/inf positions differ) | 1287 | **(a) by-zero** candidate, but 1224 (95%) also contain atan2/pow/log/reciprocal/div → mostly **inherited/shared** non-finite sources |
| max\|delta\| (numeric)              | 907  | 532 are rtol=0/atol=0 exact (integer/exact-int paths in chain); remainder ≤ fp16/fp rounding incl. **(c)** the missing-rounding fix-up |
| dtype mismatch (e.g. int64 vs float)| 186  | **(d) separate type-promotion bug**, not the floor_divide formula |
| shape mismatch                      | 88   | broadcast/out= artifacts, **(d) unrelated** |

Cleanly floor_divide-attributable by-zero cases (floor_divide is the first/only
chain op with a non-finite reason): **29**. Most of the 1287 non-finite rows are
co-attributable to other ops in the same graph (inherited), so the true
floor_divide-only by-zero population is on the order of tens, not ~1287.

Mechanism summary:
- **(a) by-zero nan/inf position diff** — REAL bug; cleanly attributable ~29
  graphs, contributing to many of the 1287 multi-op non-finite chains. This is
  the `w0:1515` mechanism.
- **(b) integer floor-vs-trunc** — **NOT a bug in this kernel** (repro matches
  eager). Any exact-int deltas in the cluster come from other ops or from
  `div.out_mode` trunc-vs-floor mode selection, not from `utils::floor_divide`.
- **(c) fp16 / fp rounding** — minor REAL artifact from the missing `floor`+0.5
  fix-up; small `max|delta|` only.
- **(d) inherited / unrelated** — the bulk: dtype-promotion (186), shape (88),
  and ~95% of non-finite chains where upstream atan2/pow/log/etc. already diverge.

## Fix
Make the float functor match ATen `div_floor_floating` exactly:
```cpp
FLOAT_T floor_divide(FLOAT_T a, FLOAT_T b) {
  if (b == 0) {
    return a / b;                                  // IEEE: 0/0->nan, x/0->±inf
  }
  const auto mod = std::fmod(a, b);
  auto div = (a - mod) / b;
  if ((mod != 0) && (std::signbit(b) != std::signbit(mod))) {
    div -= static_cast<FLOAT_T>(1);
  }
  FLOAT_T floordiv;
  if (div != 0) {
    floordiv = std::floor(div);
    if (div - floordiv > static_cast<FLOAT_T>(0.5)) {
      floordiv += static_cast<FLOAT_T>(1);
    }
  } else {
    floordiv = std::copysign(static_cast<FLOAT_T>(0), a / b);
  }
  return floordiv;
}
```
The integer overload needs no change (already correct floored division). Ideally
the portable kernel should just call the shared `c10::div_floor_floating` /
`c10::div_floor_integer` rather than maintain a divergent copy.

Regression test sketch (device-generic, eager as oracle):
```python
@parametrize("dtype", [torch.float32, torch.float16])
def test_floor_divide_by_zero(self, device, dtype):
    a = torch.tensor([0., 1., -1., 5.], device=device, dtype=dtype)
    b = torch.tensor([0., 0.,  0., 2.], device=device, dtype=dtype)
    out = run_portable(torch.floor_divide, a, b)
    self.assertEqual(out, torch.floor_divide(a, b), equal_nan=True)
    # explicit: 0.0//0.0 is nan, not +inf
    self.assertTrue(torch.isnan(out[0]))
```

## Severity & classification
**Severity: low–moderate, narrow but real.** The by-zero formula is a genuine,
source-level correctness bug: `0.0 // 0.0` (and `-0.0 // 0.0`) yield ±inf instead
of nan in the portable kernel, reproduced end-to-end on float32 and float16. It
fires whenever a zero divisor coincides with a zero dividend — uncommon in normal
models but exactly what differential fuzzing surfaces (`w0:1515`). The missing
rounding fix-up is a minor secondary divergence (tiny fp deltas).

**Honesty on scope:** the headline "2468 floor_divide MISMATCH" massively
overstates this kernel's blame. The integer floor-vs-trunc hypothesis is
**disproven** here (portable matches eager). ~95% of the non-finite mismatches
share the chain with other nan/inf-producing ops and are largely inherited; the
dtype/shape categories (274 rows) are different bugs (type promotion, broadcast).
Cleanly floor_divide-attributable by-zero divergences number in the low tens.
Classification: **REAL portable kernel bug (float by-zero), small blast radius**;
everything else in the cluster is precision/inherited/other-op artifact.

Citations: math_util.h:44-59 (bug), math_util.h:34-42 (correct int), op_floor_divide.cpp:55-87,
op_div.cpp:138-167, c10/util/generic_math.h:34-58 (ATen reference).

## Replay

Self-contained minimal-export replay: [`replay_floor_divide.py`](replay_floor_divide.py).
Exports `torch.floor_divide(a, b)` with `a=[0,1,-1,5] b=[0,0,0,2]`, runs it on the
portable ExecuTorch runtime vs eager on float32 **and** float16, and asserts the
by-zero divergence at position 0 (`0.0 // 0.0`): portable `+inf`, eager `nan`
(exits 0 iff both dtypes reproduce).

```
$ cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_floor_divide.py 2>&1 \
    | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
--- torch.float32 a=[0.0, 1.0, -1.0, 5.0] b=[0.0, 0.0, 0.0, 2.0] ---
  portable: [inf, inf, -inf, 2.0]
  eager:    [nan, inf, -inf, 2.0]
  pos0: portable=inf  eager=nan
--- torch.float16 ... ---  (same: portable inf, eager nan)
REPLAY: bug REPRODUCED
```
