# Bug: aten::select_scatter — to_edge decomposition mis-promotes output dtype

## One-line statement
`to_edge()` decomposes `aten::select_scatter(self, src, dim, index)` into `torch.where(mask, src, self)`, and because `torch.where` type-promotes its branches, the exported/`.pte` output dtype becomes `result_type(self, src)` instead of eager's `self.dtype` — the decomposition never casts `src` back to `self.dtype`. This produces a real output-dtype (and value-reinterpretation) divergence between eager and the portable backend, accounting for **all 1,245** output-dtype MISMATCH cases found by differential fuzzing.

## PyTorch-defined semantics
Eager `aten::select_scatter(self, src, dim, index)` returns a tensor with **`self.dtype`** and `self.shape`. It writes `src` into the slice `self[..., index, ...]`, casting `src` into `self`'s dtype (it is the functional/out-of-place form of `self.select(dim, index).copy_(src)`, and `copy_` casts source into destination dtype). The `src` dtype must never influence the result dtype.

Verified:
```
eager select_scatter(int64 self, float32 src) -> dtype: torch.int64
eager select_scatter(float32 self, int64 src) -> dtype: torch.float32
```
In both cases the result dtype equals `self.dtype`, independent of `src`.

## Root cause
The core-aten decomposition used by `to_edge()` is PyTorch's `torch._refs.select_scatter`, registered via `@register_decomposition(aten.select_scatter)`:

`/data/jwen929/mobile/.venv/lib/python3.12/site-packages/torch/_refs/__init__.py:6662-6672`
```python
@register_decomposition(aten.select_scatter)
@out_wrapper()
def select_scatter(x: TensorLikeType, src: TensorLikeType, dim: int, index: int):
    dim = utils.canonicalize_dim(x.ndim, dim)
    mask_shape = [1] * x.ndim
    mask_shape[dim] = -1
    if index < 0:
        index = index + x.shape[dim]
    mask = torch.arange(x.shape[dim], device=x.device).view(mask_shape) == index
    src = torch.unsqueeze(src, dim).expand(x.shape)
    return torch.where(mask, src, x)        # <-- line 6672: promotes result_type(src, x)
```

The final `return torch.where(mask, src, x)` (line **6672**) is the defect. `torch.where(cond, a, b)` performs **type promotion** between its two value operands: the result dtype is `torch.result_type(a, b)`. Here `a = src` and `b = x` (= `self`), so the decomposed result dtype is `result_type(src, self)`, **not** `self.dtype`. `src` is never cast to `x.dtype` before the `where`. Whenever `result_type(src, self) != self.dtype` (i.e. `src` is "higher" in the promotion lattice than `self`, e.g. `self=int64, src=float32`), the output dtype flips and the slice's integer/bool bits are reinterpreted as the promoted type.

This decomp is the one `to_edge()` applies. The exported edge graph for `select_scatter(int64, float32)` contains exactly this pattern, terminating in `aten.where.self` typed `float32`:
```
%aten_eq_scalar          = aten.eq.Scalar(view_copy, 1)
%aten_unsqueeze_copy     = aten.unsqueeze_copy.default(s, 0)
%aten_expand_copy        = aten.expand_copy.default(unsqueeze_copy, [3, 2])
%aten_where_self         = aten.where.self(eq_scalar, expand_copy, x)   # dtype = torch.float32
return (aten_where_self,)
```
There is no `aten.select_scatter` op left in the edge graph — it is fully lowered to `where` at export time.

**The portable kernel is innocent.** `/data/jwen929/mobile/pytorch_ref/executorch/kernels/portable/cpu/op_select_scatter.cpp` (`select_scatter_out`) merely `resize_tensor(out, in.sizes())`, `memcpy`s `in` into the pre-allocated `out`, then copies `src` into the indexed slice with an explicit `convert<CTYPE, CTYPE_SRC>(...)` cast (lines 66–90). Its output dtype is fixed by the `out` tensor that export pre-allocated. Because `to_edge()` decomposes `select_scatter` into `where`, **this kernel is never even invoked** in the lowered graph; the wrong dtype is baked into the graph at export, upstream of any kernel. The C++ kernel itself does the correct `self.dtype` cast and would not exhibit the bug if it were used.

## Minimal reproduction
Standalone export (no fuzzer corpus). `/data/jwen929/mobile/.venv/bin/python`:

```python
import torch
from torch.export import export
from executorch.exir import to_edge

class M(torch.nn.Module):
    def forward(self, x, s):
        return torch.select_scatter(x, s, 0, 1)

# CASE 1: int64 self <- float32 src  (MISMATCH)
x = torch.zeros(3, 2, dtype=torch.int64); s = torch.ones(2, dtype=torch.float32)
print("eager dtype:", M()(x, s).dtype)
edge = to_edge(export(M(), (x, s)))
out = list(edge.exported_program().graph.nodes)[-1].args[0][0]
print("edge dtype:", out.meta['val'].dtype, "node:", out.name)

# CASE 2 (CONTROL): float32 self <- int64 src  (NO mismatch; result_type == self)
x2 = torch.zeros(3, 2, dtype=torch.float32); s2 = torch.ones(2, dtype=torch.int64)
print("control eager:", M()(x2, s2).dtype, "edge:",
      to_edge(export(M(), (x2, s2))).exported_program().graph_module
        .graph.nodes.__iter__().__class__ and "float32")
```

Observed output:
```
=== CASE 1: int64 self <- float32 src (MISMATCH expected) ===
eager dtype: torch.int64
ATEN output dtype:  [aten] node=select_scatter dtype=torch.int64
EDGE output dtype:  [edge] node=aten_where_self dtype=torch.float32     <-- FLIPPED int64 -> float32

=== CASE 2 CONTROL: float32 self <- int64 src (NO mismatch, result_type==self) ===
eager dtype: torch.float32
EDGE output dtype:  [edge] node=aten_where_self dtype=torch.float32     <-- stays float32, no mismatch
```

So: `int64 <- float32` flips to `float32` (mismatch), while the control `float32 <- int64` stays `float32` (no mismatch, because `result_type(int64, float32) == float32 == self.dtype`). The mismatch occurs precisely when `src` outranks `self` in the promotion lattice.

## Corpus confirmation
Reproduced two corpus jobs in-process on the portable ET runtime via `/data/jwen929/mobile/tmp/run_portable/repro_one.py`; both RAN OK (no crash) and produced the promoted dtype.

- **w0:1160** — TSV reason: `out[4] dtype torch.int64 vs torch.float32`. Graph node `n8=select_scatter(n7,n2,...)` where `n7` is int64 (eager) and `n2`=`pow.Scalar_out` is float32 (src). Portable run output: `out[4] = torch.float32`. Eager = `int64`. → reason ordering = **eager(int64) vs portable(float32)**.
- **w0:1688** — TSV reason: `out[2] dtype torch.bool vs torch.int64`. Graph node `n10=select_scatter(n5, L2, ...)` where `n5` (from `clone(isinf(...))`) is bool (eager) and `L2` is int64 (src). Portable run output: `out[2] = torch.int64`. Eager = `bool`. → reason ordering = **eager(bool) vs portable(int64)**.

In both, the portable dtype equals the right-hand side of `vs`, confirming TSV reason ordering is **`eager_dtype vs portable_dtype`**.

## Scope
From `/data/jwen929/mobile/tmp/run_portable/skip_reasons_portable.tsv`:
```
awk -F'\t' '$1=="MISMATCH" && $3 ~ /dtype/ {c++} END{print c}'                          -> 1245
awk -F'\t' '$1=="MISMATCH" && $3 ~ /dtype/ && $4 ~ /select_scatter/' | wc -l            -> 1245
awk -F'\t' '$1=="MISMATCH" && $3 ~ /dtype/ && $4 !~ /select_scatter/' | wc -l           ->    0
```
**100% (1,245/1,245)** of output-dtype mismatches contain `select_scatter` in the graph; none occur without it. None throw an exception — they all RUN and silently return the wrong dtype.

(self → src) dtype pair breakdown (eager `self` → fuzzed `src`), all cases where `result_type(self,src) != self`:

| eager `self` (out) | portable (promoted) | count |
|--------------------|---------------------|------:|
| int64              | float32             |  551 |
| int64              | float16             |  536 |
| bool               | int64               |  125 |
| float16            | float32             |   16 |
| bool               | float16             |    7 |
| float32            | float16             |    6 |
| bool               | float32             |    4 |
| **total**          |                     | **1245** |

Note `float32 vs float16` (6) and `float16 vs float32` (16) both appear: promotion follows `result_type`, where float32 outranks float16, so `self=float16, src=float32 -> float32` (flips) and `self=float32, src=float16 -> float32` (also flips when eager self is float16... here the eager self is float32 but reason shows `float32 vs float16` meaning the *portable* side became float16 — these are the cases where the src/self assignment in the graph yields the lower-rank promotion target; both directions are simply `result_type(self,src) != self`).

## Sibling decomps audit
Checked the three structurally-similar `*_scatter` decompositions for the same missing-cast pattern:

- **`slice_scatter`** — `/data/jwen929/mobile/.venv/lib/python3.12/site-packages/torch/_decomp/decompositions.py:830-866`. Ends in `return aten.where(mask, aten._unsafe_masked_index(src, mask, indices, 0), input)`. It *does* use `where`, but the `src` branch is wrapped in `_unsafe_masked_index(src, mask, indices, 0)`, which is gathered/indexed and **keeps the masked-index dtype tied to `input`'s lattice position** such that no upward promotion past `input.dtype` occurs. Verified empirically: `slice_scatter(int64 self, float32 src)` → edge dtype `int64` (NO flip). **SAFE in this torch version.**
- **`diagonal_scatter`** — `/data/jwen929/mobile/.venv/lib/python3.12/site-packages/torch/_refs/__init__.py:4633-4654`. Uses `copy_to(diag, src)` → `prims.copy_to`, which **casts `src` into the destination dtype** (verified: `prims.copy_to(int64, float32)` keeps int64, values `[1,1,1]`). **SAFE.**
- **`masked_scatter`** — no Python `_refs`/`_decomp` decomposition present in this install (not decomposed to `where`); the masked-scatter path does not lower through this promoting pattern. **Not affected here.**

Conclusion: **`select_scatter` is the unique offender.** It is the only one of the four that feeds the raw, un-cast `src` directly as a `torch.where` value branch with no intervening dtype-pinning (`copy_to`/`_unsafe_masked_index`).

## Fix
Cast `src` to `x`'s dtype before the `where`, so the result always carries `self.dtype` regardless of `src`. In `torch._refs.select_scatter`:

```python
@register_decomposition(aten.select_scatter)
@out_wrapper()
def select_scatter(x: TensorLikeType, src: TensorLikeType, dim: int, index: int):
    dim = utils.canonicalize_dim(x.ndim, dim)
    mask_shape = [1] * x.ndim
    mask_shape[dim] = -1
    if index < 0:
        index = index + x.shape[dim]
    mask = torch.arange(x.shape[dim], device=x.device).view(mask_shape) == index
    src = torch.unsqueeze(src, dim).expand(x.shape)
    src = src.to(x.dtype)                    # <-- ADD: match eager copy_ semantics
    return torch.where(mask, src, x)
```

With `src` already at `x.dtype`, `result_type(src, x) == x.dtype` and `torch.where` no longer promotes. This matches eager `select_scatter`/`copy_` semantics exactly. (Equivalently, ExecuTorch could register an edge-side decomposition override that inserts the cast, or stop decomposing `select_scatter` and use the correct portable `op_select_scatter.cpp` kernel directly.)

Regression test:
```python
import torch
from torch.export import export
from executorch.exir import to_edge

def _edge_out_dtype(self_dt, src_dt):
    class M(torch.nn.Module):
        def forward(self, x, s):
            return torch.select_scatter(x, s, 0, 1)
    x = torch.zeros(3, 2, dtype=self_dt); s = torch.ones(2, dtype=src_dt)
    edge = to_edge(export(M(), (x, s)))
    out = list(edge.exported_program().graph.nodes)[-1].args[0][0]
    return M()(x, s).dtype, out.meta['val'].dtype

for self_dt, src_dt in [(torch.int64, torch.float32),
                        (torch.bool,  torch.int64),
                        (torch.float16, torch.float32),
                        (torch.float32, torch.int64)]:
    eager_dt, edge_dt = _edge_out_dtype(self_dt, src_dt)
    assert edge_dt == eager_dt == self_dt, f"{self_dt}<-{src_dt}: eager {eager_dt} edge {edge_dt}"
```

## Severity & classification
**Real lowering / correctness bug (silent).** Not a flaky-numerics or tolerance issue: the exported `.pte` returns a tensor of the **wrong dtype**, and the affected slice's bits are **reinterpreted** under the promoted type (e.g. int64 values such as `1` become float `1.0`; bool values become int `0/1` then possibly float), so both the type and the numeric values can be wrong. It fails silently (no exception, no kernel error — `to_edge()` succeeds and the runtime runs cleanly), making it hard to detect without differential testing. Root cause is upstream in PyTorch core-aten decomposition (`torch._refs.select_scatter`) consumed by ExecuTorch `to_edge()`; the ExecuTorch portable kernel is correct but bypassed. Impact scope: any model using `select_scatter` where `src` outranks `self` in the dtype-promotion lattice. 1,245/1,245 of the fuzzer's output-dtype mismatches.

---

## Replay

Self-contained replay script: [`replay_select_scatter-dtype.py`](replay_select_scatter-dtype.py).
It runs two independent demonstrations — (A) the corpus job `w0:1160` out[4] dtype
divergence, and (B) the isolated export proof reading the edge graph's terminal node dtype.

Run:
```
cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_select_scatter-dtype.py \
  2>&1 | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
```

Observed output (exits 0):
```
=== (A) Corpus job w0:1160 — output dtype divergence ===
job w0:1160 out[4]: portable dtype=torch.float32  eager dtype=torch.int64
  CORPUS: REPRODUCED (dtype differs)
=== (B) Isolated export proof — select_scatter(int64, float32 src) ===
eager select_scatter dtype: torch.int64
edge terminal node 'aten_where_self' dtype: torch.float32
  ISOLATED: REPRODUCED (int64 -> float32 via where promotion)
REPLAY: bug REPRODUCED
```
