# Mismatch bugs — source-level deep-dives

Per-bug root cause for the MISMATCH verdicts, traced to the exact line of portable-kernel
C++ / export decomposition / `torch._refs` source (in `pytorch_ref/` and the in-tree
`.venv` torch). Each linked file has: PyTorch-defined semantics, the offending source with
`file:line`, a reproduction (standalone where possible, else the corpus `.job` + the
`tmp/run_portable/cmp_*.py` value-diff harness), scope, and the concrete fix.

These supersede the by-*cluster* reports in the parent folder where they disagree — the
deep-dives re-verified at the source level and corrected two earlier diagnoses (see ⚠ below).

> **Every bug here ships a self-contained `replay_<slug>.py`** that triggers it and exits 0 iff
> it reproduces. See [../REPLAY.md](../REPLAY.md) for the full index + how to run them all.

## The confirmed bugs, by severity

| # | bug | exact root cause (`file:line`) | layer | scope | severity |
|---|---|---|---|---|---|
| 1 | [select_scatter dtype](select_scatter-dtype.md) | `torch._refs.select_scatter` → `return torch.where(mask, src, x)` with no `src.to(x.dtype)` — `where` type-promotes. `.venv/.../torch/_refs/__init__.py:6672` | **export decomp** | **1,245 / 1,245** dtype mismatches (100%) | **High** — silent wrong output dtype + value reinterpretation; deterministic |
| 2 | [remainder sign](remainder-sign.md) | integer overload `return a % b` omits the floored fix-up the float overload has (`rem += b`). `pytorch_ref/.../cpu/util/math_util.h:216-218` (vs float `:205-211`) | **portable kernel** | ~684 directly-flagged (+ inherited); bool inputs routed through int path | **High** — defined PyTorch semantics, portable returns `fmod` sign |
| 3 | [sum float-in/int-out](sum-cast-order.md) ⚠ | NOT the kernel. `export` functionalizes `sum.IntList_out(out=int64)` → `sum.dim_IntList(dtype=None)`, dropping the int64 acc-dtype → accumulates in fp16, casts once | **export/functionalization** | **422** float-in→int-out sum outputs (421 int64, 1 bool); `prod`/`cumsum` share the exposure | **High** — wrong integer totals; the kernel itself is correct |
| 4 | [sign(NaN)](sign-nan.md) | `op_sign.cpp` special-cases NaN but `return val_in` (the NaN) instead of `0`. `pytorch_ref/.../cpu/op_sign.cpp:46-49` | **portable kernel** | of 755 sign-on-output non-finite cases, 692 have an upstream OOB op feeding sign | **Medium** — `sign(nan)` must be 0; ±inf/±0 are correct |
| 5 | [floor_divide by-zero](floor_divide.md) ⚠ | float by-zero functor returns `signbit(a)?-inf:+inf` instead of IEEE `a/b` (`nan` at `0/0`). `pytorch_ref/.../cpu/util/math_util.h:48-52`; shared by `op_div.cpp:157` floor-mode | **portable kernel** | only ~29 cleanly attributable (most of the 691 `floor_divide` outputs are *inherited* nan/inf); integer floor is **correct** | **Low–Medium** — real but narrow; `0/0`→`+inf` not `nan` |
| 6 | [bitwise shift negative](bitwise-shift-negative.md) | raw unguarded `lhs << rhs` / `lhs >> rhs` (C++ UB for negative/≥bitwidth count). `pytorch_ref/.../cpu/pattern/bitwise_op.h:31-43` | **portable kernel (UB)** | 3,756 / 10,066 mismatches (~37%) contain a shift node | **Low for real models / High as fuzz signal** — eager clamps to 0, portable yields `1<<(|n|&63)` |

⚠ = the deep-dive **revised** the earlier cluster-level hypothesis:
- **sum** (#3): the parent [mismatch-delta-exact.md](../mismatch-delta-exact.md) M4 attributed this to a kernel accumulate-vs-cast order. Source check shows `op_sum.cpp:108-133` already casts-then-accumulates correctly; the real defect is the `dtype=None` loss during **export functionalization** of the out-variant. Fix moves to the exporter, not the kernel.
- **floor_divide** (#5): the earlier guess of integer floor-vs-trunc is **disproven** (integer functor `math_util.h:34-42` is correct floored division). The only real sub-bug is the float `0/0`→`±inf` vs `nan`.

## Additional kernel investigations (born-here non-finite)

| bug | root cause | layer | severity |
|---|---|---|---|
| [distance p=0 NaN](distance-nonfinite.md) | portable L0 cdist/pdist `diff==0?0:1` counts a NaN diff as 1 (finite); ATen `min(ceil(abs),1)` propagates NaN. `distance_util.h:46-57` | **portable kernel** | **Medium** — real, 100% of the cdist/pdist non-finite family (`p==0` only) |
| [batch_norm var=0/eps=0](normalization-nonfinite.md) | `invstd=1/sqrt(var+eps)` unguarded → `var=0,eps=0` gives `+inf` → `±inf`; ATen `InvStd` forces `invstd=0`. `op_native_batch_norm.cpp:119,296` | **portable kernel** | **Medium** — real; group/layer-norm + log_softmax in the same cluster are inherited, not bugs |
| [maxpool2d backward OOB](maxpool2d-backward.md) | unguarded `grad_input[maxindex]+=` (no bounds check), shared with ATen — wild fuzzer index → OOB write / **SIGSEGV** | **portable kernel (UB, shared)** | **robustness** — the value mismatches are arg artifacts; the real defect is the missing bounds check |

## Split-out from the cluster reports (newly separated, each with a replay)

| bug | what it is | class |
|---|---|---|
| [prod/pow int64 overflow](prod-pow-int64-overflow.md) | int64 reduction/power overflows 2^63; both backends wrap (C++ UB), bit-patterns differ | **UB both sides** — flag, not fix |
| [reduction into bool out](reduction-into-bool-out.md) | `prod.int_out` into a narrower `bool` `out=` saturates differently than eager | **Likely real** (narrowing-out) |
| [shape: aliased out= resize](shape-aliased-out-resize.md) | eager resizes an aliased `out=`/`values=`/`min=` buffer in place; portable's static planning never does and silently returns the wrong shape | **Behavioral divergence** + missing runtime shape assertion (0 kernel bugs) |
| [comparison/bool flips](comparison-bool-flips.md) | `lt/gt/eq/logical_*` flip because an upstream float op crossed the threshold — comparator is correct | **Artifact** (inherited) |
| [transcendental fp16 precision](transcendental-fp16-precision.md) | `erf/gelu/tanh/exp/sin/sigmoid` deltas that merely exceed the strict `atol=0.001` on fp16 | **Artifact** (benign precision) |
| [non-finite inherited reposition](nonfinite-inherited-reposition.md) | a legit OOB NaN (e.g. `acos`) is produced on **both** backends, then repositioned downstream | **Artifact** (harness strictness) |

Full replay index + run instructions: [../REPLAY.md](../REPLAY.md).

## Behavioral divergence (not a kernel bug — but a real, interesting discovery)

| item | what it is | where covered |
|---|---|---|
| **All 485 shape mismatches** | **Eager vs ExecuTorch differ in out-variant resize/alias semantics:** ATen out-variants (`topk`/`min.dim`/`max.dim`) resize their aliased `out=` buffer in place at runtime and the new shape propagates downstream; ExecuTorch portable uses statically planned buffers, never resizes, and **silently returns the wrong shape with no runtime check.** Not a kernel correctness bug and won't fire on a well-formed exported model, but a genuine execution-model difference + a robustness gap (missing planned-vs-produced shape assertion). The one actionable ET-side ask is that runtime assertion. | [../mismatch-shape.md](../mismatch-shape.md) |

## Not bugs (artifacts — fix the harness/generator, not ExecuTorch)

| item | what it is | where covered |
|---|---|---|
| **fp16/fp32 transcendental & reduction precision** | `erf`/`gelu`/`tanh`/`exp`/`sin`/`sigmoid`/`rsqrt`… deltas that merely exceed `atol=0.001`; ~40-45% of the float-tolerance cluster | [../mismatch-delta-float.md](../mismatch-delta-float.md) M3/M4 |
| **prod / pow int64 overflow** | both backends wrap signed-int64 overflow (C++ UB on both sides); bit-patterns differ but neither is "correct" | [../mismatch-delta-exact.md](../mismatch-delta-exact.md) M2 |
| **comparison/bool flips, view/copy/scatter deltas** | `lt`/`gt`/`eq`/`logical_*` and structural ops that **inherit** an upstream float or integer divergence — the op itself is correct | [../mismatch-delta-exact.md](../mismatch-delta-exact.md) M6 + "inherited" |

## Two cross-cutting harness recommendations (recur in nearly every deep-dive)

1. **Bucket by relative error and scale tolerance by output dtype.** Bucketing by absolute
   `|delta|` mis-sorts real `eager≈0` bugs into the "small" bucket and flags benign fp16
   rounding as large. Use `atol ≈ 4·ulp(dtype)·max(|out|)` for float outputs.
2. **Treat inherited non-finite as a propagation artifact.** When a NaN/Inf already exists in
   an *input* to the returned op (legit upstream OOB or fp16 overflow), an exact NaN/Inf
   *position* check trips on re-ordering rather than a backend defect — de-noises the
   non-finite cluster down to the real `sign`/shift kernel bugs.

## Fix priority (engineering order)

1. **select_scatter** (#1) — one line (`src = src.to(x.dtype)` in the decomp), clears 1,245 cases, upstreamable to PyTorch `_refs`.
2. **remainder** (#2) — add `rem += b` to the integer overload, clears ~684+ cases.
3. **sum/prod/cumsum dtype loss** (#3) — preserve out-variant `out.dtype` as `dtype=` through functionalization, clears ~422.
4. **sign(NaN)** (#4) — change `return val_in` → `return 0` in the NaN branch.
5. **floor_divide** (#5) — return IEEE `a/b` at `b==0` (or call `c10::div_floor_floating`).
6. **bitwise shift** (#6) — guard negative/oversize count to match eager; plus stop the generator emitting negative shift constants so this UB stops burying real bugs.
