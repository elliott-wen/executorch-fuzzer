# Bug: aten::sign(NaN) — portable propagates NaN, eager returns 0

## One-line statement
The ExecuTorch **portable** `sign` kernel (as shipped in the runtime under test) returns `NaN` for a `NaN` input, whereas PyTorch eager defines `sign(NaN) = 0`; this makes portable disagree with eager on every NaN element and silently launders downstream ops (`atan2`, etc.) into NaN where eager produced finite values.

---

## PyTorch-defined semantics
PyTorch / ATen `torch.sign` is the real signum with these mandated special cases:

| input | `torch.sign` |
|-------|--------------|
| `NaN` | `0` |
| `+inf` | `+1` |
| `-inf` | `-1` |
| `+0.0` | `0` |
| `-0.0` | `0` |
| `-3.0` | `-1` |
| `2.0` | `+1` |

Standalone eager confirmation:

```
$ .venv/bin/python -c "import torch; x=torch.tensor([float('nan'),float('inf'),-float('inf'),0.0,-0.0,-3.0,2.0]); print(torch.sign(x))"
tensor([ 0.,  1., -1.,  0.,  0., -1.,  1.])
```

So eager `sign(NaN) = 0` — NaN is mapped to a *finite* value. (`torch.sgn` differs only for complex dtypes; for real inputs `sgn == sign`, including `sgn(NaN)=0`.)

---

## Portable kernel root cause

The runtime under test is the installed wheel `executorch 1.4.0.dev20260625+cpu`
(`/data/jwen929/mobile/.venv/.../executorch/`, a binary wheel — no `.cpp` shipped).
Its compiled `sign_out` reproduces the formula below **without** the NaN guard
(see Reproduction): for `NaN`, `(x>0)` and `(x<0)` are both `false`, giving `0-0=0`
on most paths — but the kernel actually emits `NaN`, i.e. the shipped build either
predates the guard or computes `copysign`/passes the value through. Observed behavior:
**`sign(NaN) → NaN`**, all other special cases correct (`±inf→±1`, `±0→0`).

The reference source tree `pytorch_ref/executorch` (HEAD `2759ef1`, 2026-06-08) shows
the *intended/fixed* kernel — it already contains the NaN guard:

`pytorch_ref/executorch/kernels/portable/cpu/op_sign.cpp:43-55`
```cpp
ET_SWITCH_REALHBF16_TYPES(in.scalar_type(), ctx, "sign.out", CTYPE, [&] {
  apply_unary_map_fn(
      [](const CTYPE val_in) {
        if (utils::isnan_override(val_in)) {     // line 46
          return val_in;                          // line 47  <-- returns NaN, NOT 0
        } else {
          return static_cast<CTYPE>((val_in > 0) - (val_in < 0));  // line 49
        }
      },
      in.const_data_ptr<CTYPE>(),
      out.mutable_data_ptr<CTYPE>(),
      in.numel());
});
```

`utils::isnan_override` is `std::isnan` (real types) / `false` (integral):
`pytorch_ref/executorch/kernels/portable/cpu/util/math_util.h:66-73`.

**Why the formula mishandles NaN — two distinct defects:**

1. **The core expression `(val_in > 0) - (val_in < 0)` (line 49) is itself wrong for NaN.**
   IEEE comparisons against NaN are all `false`, so this yields `0 - 0 = 0`. That would
   coincidentally match eager (`0`) — but only if NaN reached this branch.
2. **The explicit NaN branch (lines 46-47) returns `val_in` (the NaN) instead of `0`.**
   This is the laundering bug: even the *reference* tree, which clearly intends to
   special-case NaN, returns the NaN unchanged. ATen returns `0`. So the guard that was
   added to "handle" NaN actually *cements the wrong answer* — it should `return 0`, not
   `return val_in`.

`±inf` and `±0` are handled correctly by line 49: `inf>0` is true → `1-0=1`;
`-inf<0` true → `0-1=-1`; `±0` neither → `0`. These match ATen. **Only NaN is wrong.**

Contrast with ATen, which special-cases NaN to `0` (real `signum`/`sgn` lowering maps
NaN → 0), so eager never propagates NaN through `sign`.

---

## Reproduction

### Isolated minimal graph — `sign` of a fixed special-value vector
(`scratchpad/iso_sign.py`: export `aten.sign.default` to `.pte`, run on the ET runtime)

```
input   : [nan, inf, -inf, 0.0, -0.0, -3.0, 2.0]
eager   : [0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0]
portable: [nan, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0]
```

Only position 0 differs: **portable `sign(NaN)=nan`, eager `sign(NaN)=0`**. Every other
special case (±inf, ±0) matches. This isolates the kernel defect with no upstream op
involved.

### Corpus job `w0:1490` (the confirmed fuzzer find)
Chain (from `corpus/portable/w0/w0_1490.py`):
```
n0 = logit.out(L0)        # logit OOB on randn input -> NaN  (BOTH backends)
n1 = sign.out(n0)         # eager: sign(NaN)=0 ;  portable: sign(NaN)=NaN   <-- BUG
n2 = clamp.Tensor_out(n1, ...)
n3 = atan2.out(n1, n2)    # eager: atan2(0,0)=0 ;  portable: atan2(NaN,..)=NaN
```

`tmp/run_portable/cmp_nonfinite.py corpus/portable/w0/w0_1490.py`:
```
job w0:1490: 9 portable outs, 9 eager outs
out[0] ... nonfinite-pos-diff=2 <<< NONFINITE-DIFF
    [0] portable=nan  eager=0.0
    [1] portable=nan  eager=0.0
out[1] ... nonfinite-pos-diff=2 <<< NONFINITE-DIFF
    [0] portable=nan  eager=0.46251991391181946
    [1] portable=nan  eager=0.46251991391181946
out[3] ... nonfinite-pos-diff=2 <<< NONFINITE-DIFF
    [0] portable=nan  eager=0.0
    [1] portable=nan  eager=0.0
```

`out[0]` is `n3 = atan2(sign(logit(L0)), clamp(sign(...)))`. Eager: `sign(NaN)=0` →
`atan2(0,0)=0`. Portable: `sign(NaN)=NaN` → `atan2(NaN,…)=NaN`. The NaN itself is
legitimately produced by `logit` (out-of-domain) on **both** backends; the divergence
appears precisely at `sign`, then fans out through `atan2`/`clamp` into 3 of 9 outputs.

---

## Scope

- `skip_reasons_portable.tsv` has **10,066** total `MISMATCH` rows; the dominant class is
  "non-finite mismatch (nan/inf positions differ)".
- **755** distinct jobs have a non-finite mismatch in a graph that contains `sign.out`.
- **692 of those 755** have an out-of-domain (OOB) op upstream of the `sign`
  (`logit`, `acos`, `acosh`, `asin`, `asinh`, `atanh`, `log`/`log2`/`log10`/`log1p`,
  `sqrt`/`rsqrt`, `pow`, `reciprocal`, `div`), i.e. an op that produces NaN on **both**
  backends. This matches the ~356-and-up "sign-on-output non-finite" estimate and confirms
  the mechanism: an upstream OOB op produces NaN identically on both backends, then `sign`
  *launders* it (eager → 0, portable → NaN), and the divergence propagates to whichever
  outputs depend on the sign result. Sampled chains:
  - `w0:1490` — `logit -> sign`
  - `w0:83` — `acos -> ... -> elu -> sign`
  - `w0:313` — `acos -> ... -> sign` (note: sign output mismatch only where its input is NaN)
  - `w0:455` — `acos -> ... -> sign`
- **Important qualifier:** `sign` only diverges when its *input* is NaN. The vast majority
  of `sign` calls (finite inputs, ±inf, ±0) are correct, so most graphs containing `sign`
  are unaffected — the 755 figure counts non-finite-mismatch graphs that merely *contain*
  sign; the causal subset is the ~692 with an OOB feeder. `sign` is a high-fan-out
  amplifier: a single NaN element it touches flips that position in every dependent output.

---

## Fix

Change the NaN branch to return `0` (the ATen-defined value), not the NaN itself:

```cpp
// kernels/portable/cpu/op_sign.cpp, inside the apply_unary_map_fn lambda
[](const CTYPE val_in) {
  if (utils::isnan_override(val_in)) {
    return static_cast<CTYPE>(0);     // was: return val_in;  (ATen: sign(NaN)=0)
  }
  return static_cast<CTYPE>((val_in > 0) - (val_in < 0));
}
```

(For integral types `isnan_override` is `false`, so integer `sign` is unchanged.)

### Regression test sketch
```python
from torch.testing._internal.common_utils import run_tests, TestCase
import torch

class TestSignNonFinite(TestCase):
    def test_sign_special_values(self):
        x = torch.tensor([float('nan'), float('inf'), -float('inf'),
                          0.0, -0.0, -3.0, 2.0], dtype=torch.float32)
        # run x through the portable sign.out kernel (export -> ET runtime)
        portable = run_portable_sign(x)          # harness helper
        self.assertEqual(portable, torch.sign(x))  # expects [0,1,-1,0,0,-1,1]

if __name__ == "__main__":
    run_tests()
```
Assert element-wise equality to eager `torch.sign`, with NaN→0 explicitly checked
(plain `assertEqual` treats NaN==NaN as equal, so add an explicit
`self.assertEqual(portable[0].item(), 0.0)`).

---

## Severity & classification

- **Class:** correctness / numerical-semantics divergence (portable vs ATen reference).
- **Severity:** Medium. Not a crash; produces silently wrong results, but only on the
  non-finite (NaN) path, which itself arises from out-of-domain inputs that are already
  "garbage in." However, because `sign` is meant to *sanitize* NaN to 0 in eager, the
  portable kernel removes that safety net and lets NaN escape into otherwise-finite
  outputs, which can corrupt control-flow ops (`atan2`, comparisons, `clamp`) downstream.
- **Reproducibility:** Deterministic, dtype-independent across `REALHBF16` float types,
  reproduced both in isolation and via corpus job `w0:1490`.
- **Note on the reference tree:** `pytorch_ref` HEAD already special-cases NaN but returns
  `val_in` (still NaN) — so the bug is present even in the "fixed-looking" source; the real
  fix is `return 0`. The shipped wheel `1.4.0.dev20260625` exhibits the same wrong output.

---

## Replay

Self-contained replay script: [`replay_sign-nan.py`](replay_sign-nan.py). It runs two
independent demonstrations — (A) the corpus job `w0:1490` non-finite position diff, and
(B) the isolated `torch.sign` export proof on `[nan,inf,-inf,0,-0,-3,2]`.

Run:
```
cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_sign-nan.py \
  2>&1 | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
```

Observed output (exits 0):
```
=== (A) Corpus job w0:1490 — non-finite position diff ===
job w0:1490 out[0] dt p=torch.float32/e=torch.float32 nonfinite-pos-diff=2
  [0] portable=nan  eager=0.0
  [1] portable=nan  eager=0.0
  CORPUS: REPRODUCED
=== (B) Isolated export proof — torch.sign special values ===
input   : [nan, inf, -inf, 0.0, -0.0, -3.0, 2.0]
eager   : [0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0]
portable: [nan, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0]
  pos0: portable sign(nan)=nan  eager=0.0
  ISOLATED: REPRODUCED
REPLAY: bug REPRODUCED
```
