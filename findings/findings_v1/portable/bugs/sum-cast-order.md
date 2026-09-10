# Bug: aten::sum (float in / int out) — accumulate-then-cast vs cast-then-accumulate

## One-line statement
`sum(float_input, dtype=None, out=int64)` differs between eager (97) and the exported
ExecuTorch program (116) because the `out=` integer dtype, which eager uses to drive a
per-element **cast-then-accumulate** in int64, is **lost during `torch.export`
functionalization** — the functional `sum.dim_IntList` with `dtype=None` accumulates in the
float **input** dtype and casts the float total to int64 once at the end
(accumulate-then-cast). Confirmed on `w0:707` out[5]: portable=116, eager=97.

---

## PyTorch-defined semantics (cast-then-accumulate when out is integral)

PyTorch's `sum` with an integer output type casts each element to the output integer type
**first** (per-element truncation toward zero) and then sums integers. The per-element
truncation residuals are dropped before they can accumulate, so the integer total is smaller
than `int(float_sum)`.

Standalone proof (`.venv/bin/python`):

```
$ .venv/bin/python -c "import torch; x=torch.tensor([0.07,5.84,6.83,4.9,3.2],dtype=torch.float16); \
    print(torch.sum(x,dtype=torch.int64), x.to(torch.int64).sum(), int(x.sum()))"
sum dtype=int64: tensor(18)        # cast-then-accumulate  (PyTorch semantics)
x.to(int64).sum(): tensor(18)      # identical: confirms cast-then-accumulate
int(x.sum()):     20               # accumulate-then-cast  (the wrong answer)
x.double().sum(): 20.8356          # the true float total, truncated -> 20
```

For the real `w0:707` data (`n0 = remainder.Scalar(L0, 7)`, float16, 36 elements):

```
eager sum(dtype=int64) cast-then-acc : 97
n0.to(int64).sum()  (cast-then-acc)  : 97
int(n0.sum())       (acc-then-cast)  : 116
n0.double().sum()                    : 116.84732055664062   # truncates to 116
```

So **eager = 97 (cast-then-accumulate)**, and **116 = `int(float_sum)` (accumulate-then-cast)**.

---

## Root cause — it is NOT the portable C++ kernel

The portable kernel is correct in isolation. The generic reduction path casts **per element**
before accumulating, accumulating in `CTYPE_OUT`:

`pytorch_ref/executorch/kernels/portable/cpu/op_sum.cpp:108-133`
```cpp
ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(),  ctx, op_name, CTYPE_IN, [&] {     // :108
  ET_SWITCH_REALHBBF16_TYPES(out.scalar_type(), ctx, op_name, CTYPE_OUT, [&] {  // :109-110
    CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
    ...
      CTYPE_OUT sum = 0;                                 // :116  accumulator is CTYPE_OUT
      sum = plan->execute<CTYPE_IN, CTYPE_OUT>(
          [](CTYPE_IN v) { return static_cast<CTYPE_OUT>(v); },   // :119-121 per-element cast
          [](CTYPE_OUT outv, CTYPE_OUT acc) { return acc + outv; },// :122-124 accumulate in CTYPE_OUT
          out_ix);
```

The map functor casts each input element to `CTYPE_OUT` (int64) **before** it enters the
`acc + outv` reduction, and `acc` is `CTYPE_OUT`. That is exactly cast-then-accumulate.
The supporting reduction helper `MapReduceOverDimListPlan::execute`
(`pytorch_ref/executorch/kernels/portable/cpu/util/reduce_util.h:542-566`) seeds
`CTYPE_OUT acc_val = map_fun(in_data[init_index])` and reduces `reduce_fun(map_fun(...), acc_val)`
— map (cast) is applied to every element, accumulator type is `CTYPE_OUT`. The same is true
for the `mean`, `prod`, and `cumsum` kernels (see Scope). The `out == in` dtype fast path at
`op_sum.cpp:49-77` does not apply here (`float16 != int64`).

**Proof the kernel is correct:** a minimal `to_edge(...).to_executorch()` of
`torch.sum(x, dtype=torch.int64)` on the **ET runtime** returns the CORRECT value:

```
input: [0.07, 5.84, 6.83, 4.9, 3.2] (float16)
eager cast-then-acc  : 18
acc-then-cast (float): 20
PORTABLE ET runtime  : 18      # <-- correct, matches eager
```

### The actual divergence: out-variant `dtype` lost in functionalization

The fuzzer's eager oracle calls the **out-variant** exactly as generated:
`torch.ops.aten.sum.IntList_out(n0, None, True, dtype=None, out=<int64 tensor>)`.
`torch.export` functionalizes that into the **functional** `sum.dim_IntList(n0, None, True)`
with `dtype=None`, dropping the int64 signal that the `out=` tensor carried. The two forms
disagree:

```
eager out-variant   sum.IntList_out(dtype=None, out int64) : 97   # uses out.dtype=int64 -> cast-then-acc
functional          sum.dim_IntList(dtype=None)            : 116.875  (float16!)  # acc in input dtype
  cast that float sum to int64                             : 116   # the exported program's answer
functional          sum.dim_IntList(dtype=int64)           : 97   # explicit dtype agrees with eager
```

So when `dtype=None`:
- the **out-variant** infers accumulation dtype from `out.scalar_type()` (int64) → 97;
- the **functional form** ignores the (now absent) out tensor, accumulates in the **input**
  float dtype, and an inserted cast truncates the float total once → 116.

This is a **functionalization / dtype-inference divergence**, not a portable-kernel arithmetic
bug. The portable kernel faithfully executes whatever dtype the exported graph hands it; the
exported graph simply hands it the wrong (float) accumulation when the original op relied on
`out.dtype` with `dtype=None`.

---

## Reproduction — w0:707 element values

`corpus/portable/w0/w0_707.py` op:
`n14 = torch.ops.aten.sum.IntList_out(n0, None, True, dtype=None, out=torch.empty((1,1,1,1), dtype=torch.int64))`
where `n0 = remainder.Scalar(L0, 7)` is float16 (verified bit-identical between portable and eager).

`n0` (36 float16 values): `5.844, 6.836, 1.307, 0.974, 0.188, 0.075, 2.527, 6.742, 0.918,
5.578, 6.203, 0.314, 6.977, 0.964, 0.029, 6.102, 5.984, 6.219, 0.672, 1.093, 0.815, 0.812,
5.828, 6.480, 0.626, 0.719, 0.059, 5.438, 6.227, 0.417, 0.857, 6.859, 1.322, 1.582, 6.465, 6.797`

Run via `tmp/run_portable/cmp_delta.py` against the stored `.pte`:

```
$ .venv/bin/python tmp/run_portable/cmp_delta.py corpus/portable/w0/w0_707.py 5
job w0:707: 7 portable outs, 7 eager outs
INPUTS: [('torch.float16',(1,3,3,4)), ('torch.float16',(2,3,1,3,4)), ('torch.int64',(4,)), ('torch.int64',(4,))]
out[5] dt p=torch.int64/e=torch.int64 shape=(1,1,1,1) max|delta|=1.9000e+01 at idx 0
   portable=116  eager=97
    [0] portable=116  eager=97
```

| quantity                                   | value | matches |
|--------------------------------------------|-------|---------|
| eager `sum.IntList_out(dtype=None,out int64)` | 97 | cast-then-accumulate (PyTorch semantics) |
| portable (exported .pte)                   | 116 | accumulate-then-cast |
| `n0.to(int64).sum()`                        | 97  | == eager (int-then-sum hypothesis) |
| `int(n0.float().sum())`                     | 116 | == portable (sum-then-int hypothesis) |
| `n0.double().sum()`                         | 116.847 | float total, truncates to 116 |

Portable's 116 is exactly `int(float_sum)`; eager's 97 is exactly `n0.to(int64).sum()`.

---

## Scope

Counts from `tmp/run_portable/skip_reasons_portable.tsv` via `tmp/run_portable/scope_sum.py`
(parses, for each MISMATCH row, the op producing the mismatched `out[k]`):

```
MISMATCH rows total: 10066
  mismatched-output IS sum.IntList_out: 439
  of those, out= dtype is INTEGER (cast-order candidate): 422   (421 int64, 1 bool)
  of those, out= dtype is FLOAT (precision, NOT cast-order):  8   (5 float32, 3 float16)
  unresolved (no py / no parse): 9
```

Note the naive `awk '$1=="MISMATCH" && $4 ~ /sum/'` count is **2364**, but that over-counts:
`/sum/` also matches `cumsum`, and matches any row whose graph merely *contains* a sum even
when the mismatched output is a different op. The precise figure where the **mismatched
output itself** is a float-in/integer-out `sum` is **422**. Of those, the genuine cast-order
cases (small integer deltas like w0:707's 19, w1:520's 1, w10:1146's 3) are distinct from the
`9.223e+18`-delta rows, which are float→int64 **overflow/saturation** cases (a related but
separate consequence of accumulating in float then casting an out-of-range total).

### Related reductions — same `dtype=None` exposure

All four reduction kernels are written cast-then-accumulate in the C++ source, so they are
correct in isolation; the exposure is the same export-time `dtype=None` functionalization:

- **prod** (`op_prod.cpp:37-44`, `:77-96`): `static_cast<CTYPE_OUT>(data_in[i])` per element,
  accumulates `data_out[0] *= ...` in `CTYPE_OUT`. Same float-in/int-out risk; appears in the
  corpus (e.g. `w0:1007`, `w0:1053`).
- **mean** (`op_mean.cpp:84-98`): casts per element to `CTYPE_OUT`, but `out` is constrained to
  a **float** type (`ET_SWITCH_FLOATHBF16_TYPES` on out), so integer-out truncation does not
  arise; differences seen for mean are ordinary float-precision, not cast-order.
- **cumsum** (`op_cumsum.cpp:38-80, 114-118`): `load_self` casts each element to `CTYPE_OUT`
  before the running add; same float-in/int-out cast-order exposure under `dtype=None`.

---

## Fix

The defect is upstream of the portable kernel: **preserve the integer accumulation dtype when
an `out=` integer tensor is functionalized with `dtype=None`.** Two equivalent fixes:

1. **Functionalization (preferred):** when lowering `sum.IntList_out` / `sum.int_out` (and
   `prod`, `cumsum`) whose `out.dtype` is integral and `dtype` arg is `None`, set the functional
   op's `dtype` to `out.dtype` so the graph carries `sum.dim_IntList(..., dtype=int64)`
   (verified above to give 97, matching eager). This is the correct ATen contract: the
   out-variant's accumulation type is `out.dtype` when `dtype` is unset.
2. **Kernel-side belt-and-suspenders:** keep accumulating in `CTYPE_OUT` (already the case) and
   ensure the exported edge graph never inserts a *trailing* float→int cast that bypasses
   per-element truncation — i.e. forbid a `sum(float)->float` followed by `_to_copy(int64)`
   when the source op had an integer `out=`.

### Regression test sketch (ExecuTorch, device-generic)

```python
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestSumCastOrder(TestCase):
    def test_sum_float_in_int_out_cast_then_accumulate(self, device):
        x = torch.tensor([0.07,5.84,6.83,4.9,3.2], dtype=torch.float16, device=device)
        # eager out-variant (dtype=None) drives accumulation off out.dtype
        ref = torch.empty(1, dtype=torch.int64, device=device)
        torch.ops.aten.sum.IntList_out(x, None, True, dtype=None, out=ref)
        # exported program must agree (cast-then-accumulate -> 18, not 20)
        got = run_exported(lambda t: torch.sum(t, dtype=torch.int64), x)
        self.assertEqual(got.item(), ref.item())   # 18, not 20
        self.assertEqual(ref.item(), x.to(torch.int64).sum().item())

instantiate_device_type_tests(TestSumCastOrder, globals())
if __name__ == "__main__":
    run_tests()
```

---

## Severity & classification

- **Class:** numerical-correctness / dtype-promotion divergence (silent wrong result, no crash).
- **Layer:** `torch.export` functionalization / edge lowering dropping the `out=` integer dtype
  when `dtype=None` — **not** a portable-kernel arithmetic bug (the kernel is correct in
  isolation; minimal export of `sum(..., dtype=int64)` returns the correct 18/97).
- **Severity:** Medium. Silent and data-dependent (only float-in/int-out reductions with
  `dtype` unset), but the magnitude can be large: small per-element residuals (w0:707: 97 vs
  116, +20%) up to full int64 overflow/saturation (`9.223e+18` deltas) when the float total
  exceeds the integer range.
- **Reach:** 422 corpus rows where a float-in/integer-out `sum` is the mismatched output;
  `prod` and `cumsum` share the same `dtype=None` exposure (`mean` does not — its out is
  constrained to float).
- **Trigger condition:** float input + integer `out=` + `dtype=None` on a reduction, lowered
  through `torch.export` → `to_edge` → `to_executorch`. Supplying an explicit `dtype=int64`
  makes eager and exported agree (both 97), confirming the dtype-loss diagnosis.

## Replay

Self-contained in-process replay: [`replay_sum-cast-order.py`](replay_sum-cast-order.py).
Loads the stored corpus job `corpus/portable/w0/w0_707.py`, runs it on the
portable ExecuTorch runtime and against the eager reference, and asserts the
documented divergence at USER_OUTPUT index 5 (exits 0 iff reproduced).

```
$ cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_sum-cast-order.py 2>&1 \
    | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
job w0:707 out[5] dt p=torch.int64/e=torch.int64 max|delta|=1.9000e+01 at idx 0
  portable=116  eager=97
REPLAY: bug REPRODUCED
```

Portable's exported program accumulates the float16 input then casts the total once
(`int(float_sum) = 116`); eager's out-variant casts per element first
(`n0.to(int64).sum() = 97`) — max|delta| = 19.
