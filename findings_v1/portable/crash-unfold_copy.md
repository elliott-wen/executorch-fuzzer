# CRASH cluster: unfold_copy (2,116 crashes)

## Summary
ExecuTorch portable `unfold_copy_out` aborts (SIGABRT, exit 134) when its input
tensor is **0-dimensional** (a scalar, `dim()==0`). The argument-validation
helper `check_unfold_copy_args` accepts `dim=0` for a 0-D tensor (via
`tensor_has_dim`, which explicitly allows `d==0` for 0-D), then immediately calls
`self.size(dim)` to bounds-check `size`. `Tensor::size(0)` on a 0-D tensor hits a
hard `ET_CHECK_MSG` (`dim < dim_ && dim >= 0`, range `[0, -1]`) and aborts the
runtime. Eager PyTorch, by contrast, treats `unfold` on a 0-D tensor as legal and
returns a 1-D tensor. This is a **real ExecuTorch portable bug**: a missing /
mis-ordered 0-D guard turns a case eager handles cleanly into a native abort.

## Reproduction
Command (one process per job):
```
cd /data/jwen929/mobile && .venv/bin/python tmp/run_portable/repro_one.py corpus/portable/w0/<job>.job 2>&1; echo "exit=$?"
```
Verified `exit=134` (SIGABRT / core dumped) for w0_104.

| job_id | abort site | offending call | input feeding unfold |
|--------|-----------|----------------|----------------------|
| w0:104 | `[tensor_impl.h:136] size(), assert failed (dim < dim_ && dim >= 0): ... range [0, -1], got 0` | `unfold_copy.default(n1, 0, 0, 6)` | `n1 = min.unary_out(...)` → **0-D** scalar, shape `()` |
| w0:1095 | same (`range [0,-1], got 0`) | `unfold_copy.default(n9, 0, 0, 8)` | `n9 = prod(...)` → 0-D scalar |
| w0:1701 | same | `unfold_copy.default(n10, 0, 0, 7)` | `n10 = rsub.Scalar(asinh(any.all_out(...)))` → 0-D scalar |
| w0:1302 | same | `unfold_copy.default(n2, 0, 0, 5)` | `n2 = expm1(sum.IntList_out(...))` → 0-D scalar |
| w0:1418 | `[reduce_util.cpp:380] dtype is 5` then same size() abort | `unfold_copy.default(n1, 0, 0, 8)` | 0-D scalar (from a reduce) |
| w0:1193 | same (reduce noise + size() abort) | `unfold_copy.default(n12, 0, 0, 8)` | 0-D scalar |
| w0:1534 | same (`range [0,-1], got 0`) | `unfold_copy.default(n7, 0, 0, 1)` | 0-D scalar |
| w0:1684 | `[tensor_impl.h:136] ... range [0, 1], got -2` (DISTINCT variant) | `unfold_copy.default(n4, 0, 2, 7)` | `n4 = acos.out(...)`, 2-D `(2,4)` — see note |

The dominant signature (7 of 8 above, and the bulk of the cluster) is the
identical `range [0, -1], got 0` abort = 0-D input. The `[reduce_util.cpp:380]
dtype is 5` line on some jobs is unrelated upstream-reduce noise, not the abort
cause; the process still dies on the same `tensor_impl.h:136` line.

### Quantification of the 2,116 jobs
All 2,116 `CRASH` jobs whose op-chain contains `unfold_copy` were scanned for the
concrete `unfold_copy.default(...)` call in their `.py`:
- **1,701 / 2,116 (80%)** pass literal `size=0` at `dim=0` — the canonical
  0-D-scalar abort path (a 0-D input always has `size==0` supplied by the
  generator because the only valid "size" the generator can pick for a 0-length
  dim is 0). This is the unambiguous primary mechanism.
- The remaining ~20% are mostly the same 0-D path with different incidental
  args, plus a small tail of the `w0:1684`-style variant.

## Root cause
`pytorch_ref/executorch/kernels/portable/cpu/op_unfold_copy.cpp:23-24`:
```cpp
ET_KERNEL_CHECK(
    ctx, check_unfold_copy_args(self, dim, size, step), InvalidArgument, out);
```
`check_unfold_copy_args` —
`pytorch_ref/executorch/kernels/portable/cpu/util/copy_ops_util.cpp:986-1007`:
```cpp
bool check_unfold_copy_args(const Tensor& self, int64_t dim, int64_t size, int64_t step) {
  if (dim < 0) { dim += nonzero_dim(self); }
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(self, dim));   // line 994: PASSES for 0-D, dim==0
  ET_CHECK_OR_RETURN_FALSE(size >= 0, ...);                // line 995-996
  ET_CHECK_OR_RETURN_FALSE(
      size <= self.size(dim),                              // line 997-998: self.size(0) ABORTS on 0-D
      ...);
  ...
}
```
- `tensor_has_dim` (`tensor_util.h:650-663`) explicitly permits `d==0 || d==-1`
  for a 0-D tensor, so the guard does **not** reject `dim=0`.
- The very next check calls `self.size(dim)` = `self.size(0)`.
  `Tensor::size` (`runtime/core/portable_type/tensor_impl.h:135-142`) does
  `ET_CHECK_MSG(dim < dim_ && dim >= 0, ...)`. For a 0-D tensor `dim_ == 0`, so
  the valid range is `[0, -1]` (empty) and `size(0)` **always aborts**.

So the kernel's own validation crashes on the input it was meant to validate: the
`size <= self.size(dim)` bound at line 998 is evaluated before any 0-D special
case. (`nonzero_dim`, `tensor_util.h:429-431`, maps 0-D→1 only for the *negative
dim* adjustment, never for the `self.size(dim)` call.) Note that even if line 998
were guarded, `get_unfold_copy_out_target_size`
(`copy_ops_util.cpp:1009-1025`) and the copy loop
(`op_unfold_copy.cpp:53-66`) also call `self.size(dim)` / `out.size(dim)` and
would need the 0-D case handled to match eager's "promote to 1-D" behavior.

### w0:1684 variant (secondary, minor)
`unfold_copy.default(n4, 0, 2, 7)` on a 2-D `(2,4)` input aborts with
`range [0, 1], got -2`. Here the input is NOT 0-D and eager succeeds
(`torch.ops.aten.unfold_copy.default(randn(2,4), 0, 2, 7)` → shape `[1,4,2]`).
The negative dim `-2` reaching `size()` indicates a separate indexing defect in
this branch, but it accounts for a small minority and shares the same root file.
It is called out for honesty, not as the primary cause.

## Eager comparison
PyTorch treats `unfold`/`unfold_copy` on a 0-D tensor as valid (it promotes the
scalar to 1-D):
```
>>> t = torch.tensor(3.0); t.dim()            # 0
>>> torch.ops.aten.unfold_copy.default(t, 0, 0, 6)
tensor([])            # shape [0]
>>> t.unfold(0, 0, 6)
tensor([])            # shape [0]
```
Eager returns a clean empty 1-D tensor — **no error, no abort**. ExecuTorch
portable should produce the same `[0]`-shaped output (or, at worst, return a
graceful `InvalidArgument` SKIP), not a native abort.

## Classification & severity
**Real ExecuTorch portable bug**, with a generator-artifact amplifier:
- **Real ET bug:** `check_unfold_copy_args` calls `self.size(dim)` on a 0-D
  tensor without a 0-D guard, hard-aborting the runtime where eager succeeds.
  This is a robustness/safety defect — untrusted/odd-shaped input crashes the
  whole executor process instead of being rejected gracefully. The standard ET
  contract for unsupported args is a graceful `ET_KERNEL_CHECK` →
  `InvalidArgument` (→ SKIP), never SIGABRT.
- **Generator artifact (amplifier):** the fuzzer frequently feeds 0-D outputs of
  reductions (`min.unary_out`, `prod`, `sum`, `max`, `any.all_out`) into
  `unfold_copy` with `size=0`. This is what makes the cluster huge (2,116), but
  the abort is still ExecuTorch's fault — the same input is well-defined in
  eager.

**Disentangling narrow_copy:** several crash graphs (e.g. w0:104, w0:1302) chain
both `unfold_copy` and `narrow_copy`. They are disentangled by the abort message
and execution order: the abort fires at `tensor_impl.h:136` from `size()`, and in
every reproduced job the `unfold_copy` node executes **before** the dependent
`narrow_copy` node (narrow consumes unfold's output, e.g. `n10 =
narrow_copy(n3, ...)` where `n3 = unfold_copy(...)`). The process dies inside
`unfold_copy_out` before `narrow_copy` runs, so these crashes are owned by
`unfold_copy`, not `narrow_copy`. The op-enrichment 3.91× lift for `unfold_copy`
is consistent with this.

**Severity:** High likelihood / Medium-High impact. ~2,116 crashes (~80% from one
crisp, deterministic 0-D path). Trivially reproducible; any model that unfolds a
scalar (or reduces-then-unfolds) aborts the runtime.

## Recommendation
In `check_unfold_copy_args` (`copy_ops_util.cpp:986`), add a 0-D guard before
`self.size(dim)`: use `nonzero_dim(self)`-aware sizing (treat 0-D as length-1 for
the bound) so the `size <= self.size(dim)` check uses 1, matching eager. Then
make `get_unfold_copy_out_target_size` and the copy loop in
`op_unfold_copy.cpp` handle the 0-D→1-D promotion (or, minimally, have the check
return `false` for 0-D so it becomes a graceful `InvalidArgument`/SKIP instead of
an abort). Separately, investigate the `w0:1684` `-2` negative-dim path as a
follow-up. Add a regression test for `unfold_copy` on a 0-D input.
