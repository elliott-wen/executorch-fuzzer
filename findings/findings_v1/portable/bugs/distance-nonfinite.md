# Investigation: _cdist_forward / _pdist_forward — non-finite divergence

## Verdict (up front)

**REAL kernel bug, isolated to the `p == 0` ("counting"/L0) norm with a NaN difference.**

The portable L0 norm in `distance_util.h` decides whether an element contributes to
the count with `diff == 0 ? 0 : 1`. For a **NaN** difference, `nan == 0` is `false`, so
the kernel counts it as a non-zero (contributes `1`) and produces a **finite** result.
ATen/eager instead computes the count as `min(ceil(abs(diff)), 1)`, which **propagates
NaN** through `ceil(nan)`/`min(nan,1)`, yielding `nan`. So:

- eager `_cdist/_pdist(p=0)` with a NaN diff → **NaN**
- portable `_cdist/_pdist(p=0)` with a NaN diff → **finite integer count**

This is exactly the "all non-finite" MISMATCH family: every mismatching job has `p == 0.0`
and a NaN flowing into the distance op; eager emits NaN, portable emits a finite count,
so the diff is flagged and is 100% non-finite (on the eager side).

It is **NOT** an artifact, **NOT** an invalid-arg case (`p=0` is valid and explicitly
checked `p >= 0`), and **NOT** inherited from upstream — the two backends genuinely
disagree on NaN handling in the p=0 map. The divergence is **born in the portable
kernel** (`distance_util.h`), not propagated from a wrong upstream input.

Other p regimes (p=0.5, 1, 2, 3, large) and Inf inputs agree exactly between eager and
portable (verified below), so they are not implicated — and in fact the fuzzer only ever
samples `p == 0.0` for these ops, so 100% of the family is this single bug.

## Isolation table

Controlled minimal export on both backends (recipe from the task), 2×2 / 3×2 inputs,
`disagree` = NaN-mismatch OR inf-mismatch OR rel-error > 1e-3. Pasted output below.

### _cdist_forward (float32)

| input case | p    | eager                    | portable                 | result        |
|------------|------|--------------------------|--------------------------|---------------|
| rand       | 0.0  | [2,2,2,2]                | [2,2,2,2]                | agree         |
| zerodiff   | 0.0  | [0,2,2,0]                | [0,2,2,0]                | agree         |
| **nan_in** | **0.0** | **[nan,nan,2,2]**     | **[2,2,2,2]**            | **DISAGREE**  |
| nan_in     | 0.5  | [nan,nan,9.899,9.899]    | [nan,nan,9.899,9.899]    | agree         |
| nan_in     | 1.0  | [nan,nan,5,5]            | [nan,nan,5,5]            | agree         |
| nan_in     | 2.0  | [nan,nan,3.606,3.606]    | [nan,nan,3.606,3.606]    | agree         |
| nan_in     | 3.0  | [nan,nan,3.271,3.271]    | [nan,nan,3.271,3.271]    | agree         |
| nan_in     | 1e30 | [nan,nan,inf,inf]        | [nan,nan,inf,inf]        | agree         |
| inf_in     | 0.0  | [2,2,2,2]                | [2,2,2,2]                | agree         |
| inf_in     | 0.5..1e30 | [inf,inf,...]       | [inf,inf,...]            | agree (all p) |

### _pdist_forward (float32)

| input case      | p    | eager           | portable        | result        |
|-----------------|------|-----------------|-----------------|---------------|
| rand            | 0.0  | [2,2,2]         | [2,2,2]         | agree         |
| zerodiff_pair   | 0.0  | [0,2,2]         | [0,2,2]         | agree         |
| **nan_in**      | **0.0** | **[nan,nan,2]** | **[2,2,2]**  | **DISAGREE**  |
| nan_in          | 0.5..1e30 | [nan,nan,…]| [nan,nan,…]     | agree         |
| inf_in          | 0.0  | [2,2,2]         | [2,2,2]         | agree         |
| inf_in          | 0.5..1e30 | [inf,inf,…]| [inf,inf,…]     | agree         |

### float16

Eager `cdist`/`pdist` raise `NotImplementedError: "cdist"/"pdist" not implemented for
'Half'`, so no float16 cases reach the differential comparison; all fuzzer cases are
float32. (Portable *does* support Half via `ET_SWITCH_FLOATHBF16_TYPES`, but eager can't
be the reference there.)

Pasted disagreeing lines:
```
torch.float32 nan_in   p=0.0   eager=[nan, nan, 2.0, 2.0] port=[2.0, 2.0, 2.0, 2.0]  <<<DISAGREE   (_cdist)
torch.float32 nan_in   p=0.0   eager=[nan, nan, 2.0]      port=[2.0, 2.0, 2.0]       <<<DISAGREE   (_pdist)
```

## Kernel root cause (quoted source, file:line)

**Portable L0 norm** —
`pytorch_ref/executorch/kernels/portable/cpu/util/distance_util.h:46-57`:
```cpp
template <typename CTYPE>
struct L0 {
  static inline CTYPE map(const CTYPE& diff, const CTYPE&) {
    return diff == 0 ? 0 : 1;          // line 49: nan == 0 is false -> returns 1 (counts NaN)
  }
  static inline CTYPE reduce(const CTYPE& agg, const CTYPE& up) { return agg + up; }
  static inline CTYPE finish(const CTYPE& agg, const CTYPE&) { return agg; }
};
```
The diff fed in is `std::abs(row_i[k] - row_j[k])` (pdist: `distance_util.h:38`; cdist:
`op_cdist_forward.cpp:92`). When an operand is NaN (or the subtraction is `inf - inf`),
`diff` is NaN, `diff == 0` is `false`, and the element is counted as `1` — a finite
contribution. The `p == 0.0` dispatch to this struct is at `distance_util.h:113-114`
(pdist) and `op_cdist_forward.cpp:103-104` (cdist).

**ATen reference (the eager target)** —
`pytorch_ref/aten/src/ATen/native/cpu/DistanceOpsKernel.cpp:92-97`:
```cpp
// Zero norm
template<typename data_t>
struct zdist_calc {
  static inline data_t map(const data_t& diff, const data_t& p) { return min(ceil(abs(diff)), 1); }   // line 94
  static inline data_t red(const data_t& agg, const data_t& up) { return agg + up; }
  static inline scalar_t finish(const scalar_t agg, const scalar_t /*p*/) { return agg; }
};
```
For a NaN diff: `abs(nan)=nan`, `ceil(nan)=nan`, `min(nan,1)=nan` (std::min returns its
first arg when the comparison is unordered → `nan`). The NaN therefore **propagates** to
the aggregate and out. Eager and portable thus disagree precisely and only on NaN at p=0.
The dispatch is identical otherwise (`DistanceOpsKernel.cpp:190-200`).

**Why only p=0 diverges:** for every other p the two backends use the *same* numeric
formulas — L1 (`map=diff`), L2 (`map=diff*diff`, `finish=sqrt`), Linf (`max`), and the
general `Lp` uses `std::pow(diff,p)` + `std::pow(agg,1/p)` in **both** portable
(`distance_util.h:85-96`) and ATen (`DistanceOpsKernel.cpp:128-134`). Those naive-pow
paths are bit-for-bit equivalent intent and propagate NaN/Inf the same way, which the
isolation confirms. So this is **not** a missing p-special-case in the general path; it is
a single semantic divergence in the **p=0 map's NaN handling**. The only difference is
that portable's `diff == 0 ? 0 : 1` is *not NaN-safe*, whereas ATen's `min(ceil(abs),1)`
is.

## Scope

- **Family size:** cdist — 274 MISMATCH jobs / 281 call-sites; pdist — 105 jobs /
  107 call-sites. **100% have `p == 0.0`** (verified by extracting the literal p arg from
  every job's `_cdist_forward.default(...)` / `_pdist_forward.default(...)` line — all
  281 / 107 are `0.0`). 100% non-finite, matching the propensity sweep
  (`_cdist_forward` 73/73, `_pdist_forward` 32/32 non-finite in the sampled set).
- **Reproduced jobs (p=0.0 confirmed in-source + in-runtime):**
  cdist `w10:129`, `w0:169`, `w13:584`, `w14:378`;
  pdist `w0:334`, `w10:1592`, `w20:537`, `w15:895`.
  Example `corpus/portable/w10/w10_129.py:45`:
  `_cdist_forward.default(n18, n2, 0.0, None)` (n18 is a broadcast of a reduction → equal
  rows / possibly NaN feed). `repro_one.py` runs the program to completion (portable emits
  a finite count where eager would emit NaN, so no crash — just a value mismatch).
- **Born-here vs inherited:** **Born here.** The NaN reaching the op is a legitimate
  fuzzer-graph value (e.g. `var`/reductions, `acos` out of domain, `0/0`); eager faithfully
  propagates it (NaN out), portable swallows it into a finite count. The divergence is
  created inside the portable L0 `map`, not inherited from a backend producing a wrong
  input — both backends receive the same NaN input; only portable mis-handles it.
- **Dtype:** float32 only in practice (eager has no Half cdist/pdist). Affects all float
  CTYPEs in the portable kernel symmetrically.

## Fix / recommendation

Make the portable L0 `map` NaN-propagating to match ATen. Minimal, local fix in
`pytorch_ref/executorch/kernels/portable/cpu/util/distance_util.h:48-50`:

```cpp
static inline CTYPE map(const CTYPE& diff, const CTYPE&) {
  // match ATen zdist: min(ceil(|diff|), 1) so NaN propagates instead of counting as 1
  return std::min(std::ceil(std::abs(diff)), static_cast<CTYPE>(1));
}
```
(`std::abs` already used in this file; `diff` arrives pre-`abs`'d from the loops, so
`ceil(diff)` alone would also work, but mirroring ATen exactly with `min(ceil(abs),1)` is
safest and self-documenting.) Equivalent guard: `return (diff != diff) ? diff : (diff==0 ? 0 : 1);`
— either restores NaN propagation.

This is the only change needed; L1/L2/Linf/Lp already agree with eager on non-finite
handling. After the fix, all 274 cdist + 105 pdist non-finite MISMATCHes in this family
should resolve (eager NaN == portable NaN). Recommend upstreaming to ExecuTorch as a
correctness fix to `distance_util.h`, with a regression test feeding a NaN row at p=0 to
both `_cdist_forward` and `_pdist_forward`.

## Replay

Self-contained minimal-export replay: [`replay_distance-nonfinite.py`](replay_distance-nonfinite.py).
Exports `torch.cdist(x1, x2, p=0.0)` (the public API lowers to `_cdist_forward`) on a
2×2-vs-2×2 input where one `x1` row carries a `nan`, runs it on the portable ExecuTorch
runtime vs eager, and asserts a position that is `nan` in eager but **finite** in portable
(exits 0 iff reproduced). The NaN-involving pairs match the documented
`eager=[nan,nan,2,2]` / `portable=[…,2,2]` shape.

```
$ cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_distance-nonfinite.py 2>&1 \
    | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
x1: [[nan, 0.0], [1.0, 1.0]]
x2: [[0.0, 0.0], [2.0, 2.0]]
eager:    [nan, nan, 2.0, 2.0]
portable: [1.0, 2.0, 2.0, 2.0]
  [0] portable=1.0  eager=nan
  [1] portable=2.0  eager=nan
REPLAY: bug REPRODUCED
```

Portable's L0 `diff==0 ? 0 : 1` counts the NaN diff as a finite `1`/`2`; eager's
`min(ceil(abs(diff)),1)` propagates the NaN.
