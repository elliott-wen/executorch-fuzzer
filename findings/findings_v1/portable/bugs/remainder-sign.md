# Bug: aten::remainder — wrong result sign (fmod semantics) + divide-by-zero

## One-line statement
ExecuTorch's portable `remainder` **integer** path computes `a % b` (C truncated
modulo, sign of the *dividend*) with **no divisor-sign correction**, so for a
negative divisor it returns the `fmod`-style result instead of ATen's
floored-remainder (sign of the *divisor*) — e.g. `remainder(1, -3)` returns `1`
where eager returns `-2`. The bug is confined to the **integer** overload
(`math_util.h:216`); the **float** overload (`math_util.h:205`) already applies
the `rem += b` fix-up and is correct for both sign and divide-by-zero. Because
ATen promotes `bool`/`int` operands to an integral compute type, bool-tensor
remainders (very common in the fuzz corpus) take the buggy integer path.

## PyTorch-defined semantics
`torch.remainder(a, b)` is the floored remainder: `r = a - floor(a/b) * b`, whose
result always matches the **sign of the divisor `b`**. `torch.fmod(a, b)` is the
truncated remainder (C `fmod`/`%`), whose result matches the **sign of the
dividend `a`**. They differ exactly when `a` and `b` have opposite signs and the
remainder is non-zero.

By-zero: integer `x % 0` **raises** (`ZeroDivisionError` / kernel InvalidArgument);
float `remainder(x, 0.0)` = **nan** (same as `fmod(x, 0.0)`).

Standalone eager confirmation:
```
$ .venv/bin/python -c "import torch; print('remainder:', torch.remainder(torch.tensor([1,2,7,-1]), -8)); print('fmod:', torch.fmod(torch.tensor([1,2,7,-1]), -8))"
remainder: tensor([-7, -6, -1, -1])      # sign of divisor (-8)
fmod:      tensor([ 1,  2,  7, -1])      # sign of dividend
```
So `remainder` follows the divisor's sign, `fmod` the dividend's. Float by-zero:
```
$ .venv/bin/python -c "import torch; print(torch.remainder(torch.tensor([1.,2.,7.]),0.0), torch.fmod(torch.tensor([1.,2.,7.]),0.0))"
tensor([nan, nan, nan]) tensor([nan, nan, nan])
```

## Portable kernel root cause
Driver: `pytorch_ref/executorch/kernels/portable/cpu/op_remainder.cpp`. Both the
`.Tensor_out` (line 70) and `.Scalar_out` (line 138) overloads delegate the
per-element math to `utils::remainder_override`:
```cpp
op_remainder.cpp:70    value = utils::remainder_override(val_a, val_b);
op_remainder.cpp:138   return utils::remainder_override(val_a, val_b);
```
The functor lives in `pytorch_ref/executorch/kernels/portable/cpu/util/math_util.h`.
Two overloads, selected by compute type:

Float overload — **correct** (applies the divisor-sign fix-up):
```cpp
math_util.h:205  CTYPE remainder_override(CTYPE a, CTYPE b) {
math_util.h:206    float rem = std::fmod(a, b);
math_util.h:207    if (((a < 0) ^ (b < 0)) && rem != 0) {
math_util.h:208      rem += b;
math_util.h:209    }
math_util.h:210    return rem;
math_util.h:211  }
```
This is exactly `a - floor(a/b)*b`: `std::fmod` gives the dividend-sign remainder,
and adding `b` when the operands have opposite signs converts it to the
divisor-sign remainder. Correct.

Integer overload — **BUG** (no fix-up):
```cpp
math_util.h:216  CTYPE remainder_override(CTYPE a, CTYPE b) {
math_util.h:217    return a % b;
math_util.h:218  }
```
C `%` truncates toward zero, so its result has the sign of the dividend `a`
(identical to `fmod`). The float overload's `if (((a<0)^(b<0)) && rem!=0) rem += b`
correction is **absent here**. That single missing branch is the entire bug: for a
negative divisor (or, symmetrically, negative dividend with positive divisor) the
returned value carries the dividend's sign instead of the divisor's.

Divide-by-zero handling (correct in both kernels):
- Integer: `op_remainder.cpp:64-68` (`.Tensor`) sets `div_by_zero_error` and
  `op_remainder.cpp:108-114` (`.Scalar`) does an `ET_KERNEL_CHECK_MSG` —
  both return `InvalidArgument` ("integer division by zero"), never executing
  `a % b`. Matches eager's raise.
- Float: hits the float overload, `std::fmod(x, 0.0) == nan`, matching eager. The
  `is_integral_type` guard at `op_remainder.cpp:64` is false for floats, so no
  error is raised. Confirmed by repro below (portable returns nan, not a finite
  wrong value).

## Reproduction
Real corpus jobs whose flagged `remainder` operand is bool/int (integer path):
- `corpus/portable/w0/w0_452.py:45` — `remainder.Scalar(n13, -3)`, n13 = `logical_not(...)` (bool 0/1).
- `corpus/portable/w0/w0_1607.py:51` — `remainder.Scalar(n2, -8)`, n2 = `logical_or(...)` (bool 0/1).
- `corpus/portable/w0/w0_1143.py:32` — `remainder.Scalar(n0, -2)`, n0 = `ge.Scalar(...)` (bool).
- `corpus/portable/w0/w0_141.py:45` — `remainder.Scalar(n7, 0)` on a float tensor (by-zero, float path).

Element-level whole-graph diff (`tmp/run_portable/cmp_delta.py`), w0:452 out[1]:
```
out[1] dt p=torch.int64/e=torch.int64 shape=(2,) max|delta|=3.0000e+00 at idx 1
   portable=1  eager=-2          # remainder.Scalar(1, -3): portable==fmod(1,-3)=1, eager=-2
```

Isolated single-op portable PTE (built via `gen.export_et.build_job(..., backend="portable")`,
run on the in-process ExecuTorch runtime) — proves portable == fmod, not remainder:
```
== int remainder.Scalar(a,-3)  [w0:452]
  input=[0, 1]   portable=[0, 1]   eager=[0, -2]   fmod=[0, 1]      portable==fmod? True  ==eager? False
== int remainder.Scalar(a,-8)  [w0:1607]
  input=[0,1,7,-1] portable=[0,1,7,-1] eager=[0,-7,-1,-1] fmod=[0,1,7,-1]  portable==fmod? True  ==eager? False
== bool remainder.Scalar(a,-2)  [w0:1143]
  input=[F,T,T,F]  portable=[0,1,1,0]  eager=[0,-1,-1,0]  fmod=[0,1,1,0]   portable==fmod? True  ==eager? False
== float remainder.Scalar(a,-2.0)  [float path, control]
  input=[1,3,7,-1] portable=[-1,-1,-1,-1] eager=[-1,-1,-1,-1] fmod=[1,1,1,-1]  portable==eager? True
== float remainder.Scalar(a, 0.0)  DIVIDE-BY-ZERO [w0:141]
  input=[1,2,7]    portable=[nan,nan,nan]  eager=[nan,nan,nan]   portable==eager? True
```
Conclusion: the integer path returns `fmod` (dividend sign); the float path is
correct for both sign and by-zero. The mismatch is therefore the **integer
overload only**, triggered whenever the common compute type is integral — which
includes the very common `bool`-tensor operand case.

## Scope
`awk -F'\t' '$1=="MISMATCH" && $4 ~ /remainder/' tmp/run_portable/skip_reasons_portable.tsv | wc -l`
→ **3508** mismatch rows whose graph contains a `remainder` op.

Heuristic split (classify by whether the flagged `out[N]` producer is itself a
remainder node):
- **~729** rows: the directly-flagged mismatching output *is* a remainder node —
  the genuine kernel-bug surface.
  - **~684** numeric/sign-bug rows (integer path, negative divisor → fmod sign).
  - **~45** non-finite rows. These are **not** a remainder by-zero kernel bug
    (the isolated float by-zero repro returns nan == eager); they are wrong
    integer-path values feeding nan/inf-producing downstream ops, or nan/inf
    position differences inherited from earlier nodes.
- **~2779** rows: the flagged output is a *different* op and `remainder` only
  appears elsewhere in the graph → **inherited** divergence (a different bug, or
  a buggy remainder value propagated through later ops), not directly the
  remainder kernel under test.

(The 729/2779 split is approximate — `out[N]` indexes USER_OUTPUTs while the
heuristic indexes starred nodes — but it fixes the order of magnitude: the
direct remainder-sign bug accounts for roughly several hundred rows; there is
**no** distinct remainder divide-by-zero kernel bug.)

## Fix
Add the divisor-sign fix-up to the integer overload, mirroring the float one. In
`pytorch_ref/executorch/kernels/portable/cpu/util/math_util.h:216-218`:
```cpp
template <typename CTYPE,
          typename std::enable_if<std::is_integral<CTYPE>::value, int>::type = 0>
CTYPE remainder_override(CTYPE a, CTYPE b) {
  CTYPE rem = a % b;                       // C trunc modulo: sign of a
  if (rem != 0 && ((rem < 0) != (b < 0))) // result must match sign of b
    rem += b;
  return rem;
}
```
Guarding on `(rem < 0) != (b < 0)` (instead of `(a < 0) ^ (b < 0)`) is exactly
equivalent here and slightly more robust. `bool` reaches this path promoted to an
integer compute type, so it is covered. The integer by-zero guard in
`op_remainder.cpp` stays as-is (it already short-circuits before this functor).
Float overload and the by-zero handling need no change.

Regression test sketch (device-generic, runs eager vs portable):
```python
@parametrize("dtype", [torch.int32, torch.int64])
def test_remainder_negative_divisor_sign(self, device, dtype):
    a = torch.tensor([0, 1, 2, 7, -1, -7], dtype=dtype, device=device)
    for b in (-3, -8, 3, 8):
        out = torch.remainder(a, b)
        # floored remainder: result has sign of divisor (or is zero)
        self.assertEqual(out, a - torch.div(a, b, rounding_mode="floor") * b)
        nz = out != 0
        self.assertTrue(((out[nz] < 0) == (b < 0)).all())

def test_remainder_int_by_zero_raises(self, device):
    with self.assertRaises(RuntimeError):
        torch.remainder(torch.tensor([1, 2], device=device), 0)

def test_remainder_float_by_zero_nan(self, device):
    out = torch.remainder(torch.tensor([1., 2., 7.], device=device), 0.0)
    self.assertTrue(torch.isnan(out).all())
```

## Severity & classification
- **Class:** correctness (silent wrong value), portable CPU kernel.
- **Op:** `aten::remainder.Scalar`, `aten::remainder.Tensor` — integer/bool
  compute types only.
- **Trigger:** integral (including bool-promoted) operands with operands of
  opposite sign (negative divisor, or negative dividend with positive divisor)
  and non-zero remainder.
- **Severity:** Medium–High. Silent (no crash, no error), data-dependent, and the
  integral case is common; bool→int remainders are frequent in the corpus.
  Off-by-`b` errors in index/hashing/wrap-around math can corrupt downstream
  results without any signal.
- **By-zero:** correctly handled (integer raises, float returns nan) — **not** a
  bug; the prior "float cluster returning a finite wrong value" was the bool→int
  sign bug, not a by-zero kernel defect.
- **Backend scope:** portable reference kernel; delegated backends (xnnpack,
  etc.) lower `remainder` separately and may or may not share the defect.
```

## Replay

Self-contained in-process replay: [`replay_remainder-sign.py`](replay_remainder-sign.py).
Loads the stored corpus job `corpus/portable/w0/w0_452.py`, runs it on the
portable ExecuTorch runtime and against the eager reference, and asserts the
documented divergence at USER_OUTPUT index 1 (exits 0 iff reproduced).

```
$ cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_remainder-sign.py 2>&1 \
    | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
job w0:452 out[1] dt p=torch.int64/e=torch.int64 max|delta|=3.0000e+00 at idx 1
  portable=1  eager=-2
REPLAY: bug REPRODUCED
```

Portable returns the `fmod` result `1` (sign of dividend); eager floored
`remainder(1, -3) = -2` (sign of divisor) — max|delta| = 3.
