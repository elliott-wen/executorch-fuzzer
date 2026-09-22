# Bug: bitwise_left_shift / bitwise_right_shift — negative/oversize shift count (UB vs eager clamp)

**One-line statement:** The ExecuTorch portable kernels for `aten::bitwise_left_shift` / `bitwise_right_shift` compute a raw C++ `a << b` / `a >> b` with **no guard on the shift count**, so a negative (or `>= bitwidth`) count is C++ undefined behavior — on x86 the count is masked (`& 63`), producing a power-of-two "magic number" `1 << (|count| & 63)` instead of eager CPU's defined clamp-to-0 (left) / sign-saturation (right). When that magic int is stored into a float `out=` or fed to a float op (`cosh`, `bmm`, `expm1`) it overflows to ±Inf.

---

## PyTorch-defined (eager CPU) semantics

PyTorch's CPU shift kernels **define** out-of-range shift counts (they do not invoke C++ UB):

- **Left shift** by a count that is **negative** or **>= bitwidth** → result is **0**.
- **Right shift**: arithmetic shift that **saturates** — by a negative/oversize count the result is **0** for a non-negative operand and **-1** for a negative operand (sign fill).

Standalone confirmation (`.venv/bin/python`):

```
$ .venv/bin/python -c "import torch; x=torch.tensor([1,2,-3],dtype=torch.int64); \
    print('lshift_neg', torch.bitwise_left_shift(x,-3)); \
    print('rshift_neg', torch.bitwise_right_shift(x,-2)); \
    print('1<<61', 1<<61)"
lshift_neg tensor([0, 0, 0])
rshift_neg tensor([ 0,  0, -1])      # 1>>-2=0, 2>>-2=0, (-3)>>-2=-1  (sign saturation)
1<<61 2305843009213693952
```

So eager left-shift-by-negative = **0**, eager right-shift-by-negative saturates to 0 / -1 by sign, and `1<<61` = `2305843009213693952` — which is exactly the portable "magic number" below.

---

## Portable kernel root cause

The op TUs are thin dispatchers onto the shared bitwise pattern:

- `pytorch_ref/executorch/kernels/portable/cpu/op_bitwise_left_shift.cpp:22` and `:34` → `internal::bitwise_tensor_out<internal::bit_lshift, ...>` / `bitwise_scalar_out<internal::bit_lshift, ...>`.
- `pytorch_ref/executorch/kernels/portable/cpu/op_bitwise_right_shift.cpp:22` and `:34` → same with `internal::bit_rshift`.

The actual shift is the unguarded functor in
`pytorch_ref/executorch/kernels/portable/cpu/pattern/bitwise_op.h:31-43`:

```cpp
// bitwise_op.h:31-43
template <typename T = void>
struct bit_lshift {
  constexpr T operator()(const T& lhs, const T& rhs) const {
    return static_cast<T>(lhs << rhs);     // <-- no check on rhs<0 or rhs>=bitwidth
  }
};

template <typename T = void>
struct bit_rshift {
  constexpr T operator()(const T& lhs, const T& rhs) const {
    return static_cast<T>(lhs >> rhs);     // <-- same: raw C++ shift, UB for bad rhs
  }
};
```

`lhs << rhs` / `lhs >> rhs` where `rhs` is negative or `>= width(T)` is **undefined behavior** in C++. The compute type is selected by `utils::get_compute_type(common_type)` over `ET_SWITCH_INT_TYPES_AND(Bool, ...)` (`bitwise_op.h:96-99,139-142`); for an int64 compute type x86 lowers the shift to an instruction that masks the count by `& 63`, so a count of `-3` becomes `61`, `-4`→`60`, `-7`→`57`, etc., yielding `1 << (|count| & 63)`. There is no clamp-to-0 path anywhere — eager's defined behavior is simply absent.

---

## Reproduction

Portable runtime values via `tmp/run_portable/cmp_delta.py` (portable vs eager, element-level):

### `corpus/portable/w0/w0_1066.py` — exact-int, into int64 out
Graph node: `n4 = bitwise_left_shift.Tensor_Scalar(n1, -3)`.

```
out[0] dt p=torch.int64/e=torch.int64 shape=(1, 2, 2, 4, 2) max|delta|=2.3058e+18 at idx 0
   portable=2305843009213693952  eager=0
    [0] portable=2305843009213693952  eager=0
    ... (all 16 elements identical)
```

Portable = `2305843009213693952` = `1 << 61` = `1 << ((-3) & 63)`. Eager = `0`. **Match: magic number == `1 << (|count| & 63)`.**

### `corpus/portable/w0/w0_756.py` — into float32 out
Graph node: `n0 = bitwise_left_shift.Tensor_Scalar_out(L0, -4, out=float32)`, `L0` is `bool`.

```
out[0] dt p=torch.float32/e=torch.float32 shape=() max|delta|=5.7646e+17 at idx 0
   portable=5.764607523034235e+17  eager=0.0
```

Portable = `5.764607523034235e+17` = `1 << 59` = `576460752303423488` (the bool→int compute element shifted, then stored into a float32 `out`). Eager = `0.0`. The huge power-of-two lands in a float-declared output exactly as predicted.

### `corpus/portable/w0/w0_116.py` — fp16 overflow knock-on via `cosh`
Graph: `n8 = bitwise_left_shift.Tensor_Scalar_out(n5, -3, out=float32)` then `n10 = cosh.out(n8, out=float16)`.

```
out[1] dt p=torch.float16/e=torch.float16 shape=(2,) max|delta|=inf at idx 0
   portable=inf  eager=1.0
    [0] portable=inf  eager=1.0
    [1] portable=inf  eager=1.0
```

Portable shift produces `1<<61`-class magic int; `cosh(2.3e18)` overflows fp16 → **Inf**. Eager shift is 0, `cosh(0)=1.0`. Confirmed standalone: `torch.cosh(torch.tensor([float(1<<61)],dtype=torch.float16)) == inf`.

### Tensor-operand variant
`corpus/portable/w3/w3_2344.py:28`: `bitwise_left_shift.Tensor_out(n0, n0)` where `n0 = round.out(L0)` over int leaves generated with `lo=-4` — the **shift-count tensor itself contains negative values**, triggering the identical UB without any literal constant.

---

## Scope (both delta clusters + non-finite)

From `tmp/run_portable/skip_reasons_portable.tsv` (totals: 37746 SKIP, 10066 MISMATCH, 5996 CRASH):

```
$ awk -F'\t' '$1=="MISMATCH" && /bitwise_left_shift|bitwise_right_shift/' \
      tmp/run_portable/skip_reasons_portable.tsv | wc -l
3756
```

- **3756** MISMATCH jobs have a bitwise shift in their graph (≈ **37%** of all 10066 mismatches).
  - left-shift present: **3013**; right-shift present: **1179** (overlap where both appear).
- The non-finite (±Inf) sub-cluster (e.g. `w0:116`) is a downstream consequence of the same magic-int → float-op overflow.

Constant vs tensor count split (the TSV `graph` column abbreviates scalars as `·`, so true counts come from the `.py` corpus, 118025 graphs total):

```
$ grep -rlE 'bitwise_(left|right)_shift\.Tensor_Scalar[^)]*,[ ]*-[0-9]' corpus/portable | wc -l
8899      # graphs with a literal NEGATIVE-constant shift count
$ grep -rlE 'bitwise_(left|right)_shift\.Tensor_out' corpus/portable | wc -l
7047      # graphs with a tensor-operand shift count (may hold negatives, lo=-4)
```

- **Negative-CONSTANT count** (`.Tensor_Scalar` with literal `-N`, e.g. w0:1066 `-3`, w0:1388 `-2`, w0:756 `-4`, w0:1264 `-7`, w0:383 `-6`, w1:1363): **8899** corpus graphs.
- **Negative-TENSOR count** (`.Tensor_out`, e.g. w3:2344): **7047** corpus graphs — the count tensor can be negative because integer leaves are drawn from `[-4,5)`.

Both forms exercise the same unguarded functor.

---

## Fix

1. **Kernel (recommended): match eager's defined behavior.** In `bit_lshift` / `bit_rshift` (`bitwise_op.h:31-43`), guard the count before the raw shift:
   - Left shift: if `rhs < 0 || rhs >= bitwidth(T)` → return `0`.
   - Right shift: if `rhs < 0 || rhs >= bitwidth(T)` → return `lhs < 0 ? -1 : 0` (arithmetic sign saturation), matching PyTorch CPU.
   This eliminates the UB and makes portable bit-exact with eager. Alternatively the kernel could `ET_KERNEL_CHECK`/error on an out-of-range count, but that diverges from eager (which never errors) and would still fail the differential test — **clamp is preferred**.

2. **Generator hygiene:** the fuzzer should stop emitting **negative literal shift constants** (and should not feed potentially-negative tensors as a shift count) for `bitwise_*_shift`. No real exported model shifts by a negative amount; ~8899 corpus graphs carry this artifact and ~37% of all mismatches involve a shift node. Removing negative shift inputs separates genuine kernel divergences from self-inflicted UB noise and will sharply shrink the mismatch cluster, surfacing any *other* shift bugs that are currently buried.

Do both: (1) is the real ExecuTorch correctness fix; (2) cleans the corpus so the signal is interpretable.

---

## Severity & classification

- **Class:** Correctness divergence rooted in **C++ undefined behavior** on the portable side vs a **defined clamp/saturation** on the eager side. Not a crash (no abort), but silent wrong results, including ±Inf when the magic int reaches a float op/output.
- **Severity:** **Low–Medium for real workloads** (no production model shifts by a negative/oversize constant), but **High as a fuzzing signal** — it is the single largest contributor across the two mismatch clusters (~37% of mismatches touch a shift node) and pollutes the non-finite cluster. It is also a true latent UB in shipped kernel code: any model or pass that ever produces an out-of-range shift count gets nondeterministic, target-dependent garbage (the `& 63` masking is x86-specific; ARM masks differently, so results differ across backends/targets).
- **Recommendation:** Fix the kernel to clamp/saturate (matches eager, kills the UB), and scrub the generator so the remaining shift mismatches, if any, point at real bugs.

### Cited sources
- `pytorch_ref/executorch/kernels/portable/cpu/op_bitwise_left_shift.cpp:22,34`
- `pytorch_ref/executorch/kernels/portable/cpu/op_bitwise_right_shift.cpp:22,34`
- `pytorch_ref/executorch/kernels/portable/cpu/pattern/bitwise_op.h:31-43` (unguarded `<<` / `>>`), `:96-99,139-142` (compute-type dispatch)
- Repro jobs: `corpus/portable/w0/w0_1066.py`, `corpus/portable/w0/w0_756.py`, `corpus/portable/w0/w0_116.py`, `corpus/portable/w3/w3_2344.py`
- Tooling: `tmp/run_portable/cmp_delta.py`, `tmp/run_portable/skip_reasons_portable.tsv`

## Replay

Self-contained in-process replay: [`replay_bitwise-shift-negative.py`](replay_bitwise-shift-negative.py).
Loads the stored corpus job `corpus/portable/w0/w0_1066.py`, runs it on the
portable ExecuTorch runtime and against the eager reference, and asserts the
documented divergence at USER_OUTPUT index 0 (exits 0 iff reproduced).

```
$ cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_bitwise-shift-negative.py 2>&1 \
    | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
job w0:1066 out[0] dt p=torch.int64/e=torch.int64 max|delta|=2.3058e+18 at idx 0
  portable=2305843009213693952  eager=0
REPLAY: bug REPRODUCED
```

Portable returns `2305843009213693952 = 1 << 61 = 1 << ((-3) & 63)` (x86 shift-count
masking UB); eager clamps the negative left-shift to `0` — max|delta| ≈ 2.3058e+18.
