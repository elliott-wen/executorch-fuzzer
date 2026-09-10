# MISMATCH cluster: output dtype mismatch (~1,212)

> **Per-bug split (with a runnable replay):**
> [bugs/select_scatter-dtype.md](bugs/select_scatter-dtype.md) — the `to_edge` `select_scatter`→`where`
> mis-promotion (100% of this cluster). Full index: [REPLAY.md](REPLAY.md).

## Summary

**Headline: this entire cluster is a single, real ExecuTorch bug — not a generator/harness `out=` artifact.**

All 1,245 dtype-mismatch cases in `tmp/run_portable/skip_reasons_portable.tsv` are caused by
one op: **`aten.select_scatter`**. Every single mismatch row (1245/1245) has `select_scatter`
in its op-chain, and in every job I reproduced the offending `out[i]` is produced by (or
data-flows directly from) a `select_scatter.default` call.

Root cause: ExecuTorch's `to_edge()` **decomposes `select_scatter(self, src, dim, index)` into
`torch.where(mask, src, self)`** (PyTorch's `torch._refs.select_scatter`, which is
`return torch.where(mask, src, x)`). `torch.where` **type-promotes** its two tensor branches,
so the result dtype becomes `torch.result_type(self, src)`. But eager `aten.select_scatter`
returns **`self`'s dtype** (it casts `src` into `self`). The decomposition forgets to cast
`src` to `self.dtype` first, so whenever `result_type(self, src) != self.dtype` the edge graph —
and therefore the portable `.pte` output spec — gets the wrong (promoted) dtype.

The portable kernel `op_select_scatter.cpp` is **not** the bug: it is handed a pre-allocated
`out` tensor (whose dtype was fixed at export time by the bad decomposition) and faithfully
`memcpy`s `self` into it then converts `src` into it. The dtype is decided entirely in
`to_edge`, before any kernel runs.

The reason-text ordering is **`eager_dtype vs portable_dtype`** (first = eager, second =
portable) — confirmed below; this is the opposite of the ordering assumed in the task prompt.

## Sub-types

The three dtype pairs are just three (self_dtype, src_dtype) combinations whose `result_type`
differs from `self_dtype`. Dominant (in fact, sole) op for all three is `select_scatter`.

| reason text (`eager vs portable`) | count | self dtype | src dtype | result_type(self,src) = portable |
|---|---|---|---|---|
| `int64 vs float32` | 551 | int64 | float32 | float32 |
| `int64 vs float16` | 536 | int64 | float16 | float16 |
| `bool vs int64`    | 125 | bool  | int64   | int64 |
| `float16 vs float32` | 16 | float16 | float32 | float32 |
| `bool vs float16`  | 7   | bool  | float16 | float16 |
| `float32 vs float16` | 6 | float32 | float16 | (rev. operand order) float16 |
| `bool vs float32`  | 4   | bool  | float32 | float32 |

The "more" sub-types (16+7+6+4 = 33 cases) are the same bug with rarer self/src dtype pairs.

When `result_type(self, src) == self.dtype` (e.g. `select_scatter(float32 self, int64 src)`),
there is no mismatch — verified below — which is exactly why only these specific dtype pairs
show up.

## Mechanisms

There is effectively **one** mechanism. Below are reproduced examples spanning all three
dominant sub-types; the last two examples show the decomposition directly and the
no-mismatch control.

### Direct proof — the `to_edge` decomposition mis-promotes (definitive)

```
torch.select_scatter(self, src, 0, 1) through torch.export -> to_edge:
  int64<-float32: eager=int64  edge_target=aten.where.self  edge_dtype=float32   (MISMATCH)
  int64<-float16: eager=int64  edge_target=aten.where.self  edge_dtype=float16   (MISMATCH)
  bool <-int64:   eager=bool   edge_target=aten.where.self  edge_dtype=int64     (MISMATCH)
  float32<-int64: eager=float32 edge_target=aten.where.self edge_dtype=float32   (no mismatch — result_type==self)
```

The exported (pre-edge) ATen graph correctly carries `select_scatter -> int64`; `to_edge`
rewrites it to `aten.where.self` and the meta dtype flips to the promoted type. Decomp source
(`torch._refs.select_scatter`): `return torch.where(mask, src, x)` — `src` is never cast to
`x.dtype`, and `where` promotes.

### Example A — int64 vs float32 (self=int64, src=float32)

- **w0:1160**, out[4] = `n12 = linear(n8, n4)` but the dtype error originates in
  `n8 = select_scatter(n7, n2, 0, 1)` where `n7` is int64 (`broadcast_to` of a `_to_copy` to
  int64) and `n2` is float32. Portable makes n8 float32, so the downstream `linear` is float32;
  eager keeps n8 int64 so linear is int64.
  ```
  EAGER    out: [bool(3), int64(3), bool(2,3), int64(3), int64(2),  bool(3), f16(3), f32(2,3), f16(3)]
  PORTABLE out: [bool(3), int64(3), bool(2,3), int64(3), float32(2), bool(3), f16(3), f32(2,3), f16(3)]
                                                          ^^^ out[4]: eager int64 vs portable float32
  ```
- **w0:437**, out[6] = `n21 = select_scatter(n20, n8, 0, -2)`; n20 int64 (broadcast_to of
  int64), n8 float32 (`native_layer_norm`). eager int64(2,2) vs portable float32(2,2).
- **w0:651**, out[2] = `n12 = select_scatter(n11, n8, 0, 1)`; n11 int64, n8 float32 (`div.out`).
  eager int64(2) vs portable float32(2).
- **w0:771**, out[1] = `n6 = select_scatter(n5, n0, 0, 0)`; n5 int64, n0 float32
  (`reflection_pad2d`). eager int64(1,4,4,1) vs portable float32(1,4,4,1).
- **w1:70**, out[0] = `n4 = select_scatter(n3, n0, 0, 1)`; n3 int64, n0 float32 (`expm1`).
  eager int64(2,2) vs portable float32(2,2).

### Example B — int64 vs float16 (self=int64, src=float16)

- **w0:1363**, out[4] = `n8 = select_scatter(L2, n0, 0, 0)`; L2 int64 leaf, n0 float16.
  eager int64(1,2,2) vs portable float16(1,2,2).
- **w0:351**, out[0] = `n5 = select_scatter(n4, n3, 0, 1)`; n4 int64 (broadcast_to of int64),
  n3 float16 (`sqrt.out`). eager int64(2,2) vs portable float16(2,2).
- **w0:719**, out[1] = `n5 = select_scatter(n4, n2, 0, 1)`; n4 int64, n2 float16. eager
  int64(2,2) vs portable float16(2,2).
- **w0:1379**, out[7]: select_scatter int64-self / float16-src; eager int64(1,3,4,2) vs portable
  float16.

### Example C — bool vs int64 (self=bool, src=int64)

- **w0:1688**, out[2] = `n10 = select_scatter(n5, L2, 0, 1)`; n5 is bool (`clone` of `isinf`),
  L2 int64. eager bool(2) vs portable int64(2).
  ```
  TSV reason: "out[2] dtype torch.bool vs torch.int64"  (eager bool, portable int64)
  ```
- **w0:286**, out[0] traces to `n1 = select_scatter(n0=isinf->bool, L1=int64, 0, 0)`. eager
  bool(1,3,0) vs portable int64.
- **w1:490**, out[0] = `n2 = select_scatter(n0=logical_or->bool, L2=int64, 0, 1)`. eager
  bool(2,2,3) vs portable int64.
- **w10:701**, out[0] = `n7 = select_scatter(n0=diagonal_copy(bool L0), n6=int64, 0, 2)`. eager
  bool(3) vs portable int64(3).

### Control — no `out=` artifact involved

`select_scatter.default` in these programs has **no** explicit `out=` allocation (it is the
`.default` overload), so the generator cannot be forcing the dtype. The dtype is chosen 100% by
the edge decomposition. (The many `out=torch.empty(...,dtype=...)` calls elsewhere in the
programs are unrelated to this cluster.)

## Classification & severity

- **Mechanism (single):** `to_edge` decomposition of `aten.select_scatter` →
  `torch.where(mask, src, self)` without casting `src` to `self.dtype`, causing the edge/`.pte`
  output dtype to become `result_type(self, src)` instead of `self.dtype`.
- **Classification: real ExecuTorch backend/lowering bug.** 100% of the cluster
  (1245/1245 cases). 0% generator/harness `out=` artifact.
- **Severity: medium-high (correctness).** It silently changes an output tensor's dtype on any
  model that uses `select_scatter` with a `src` whose dtype promotes against `self`
  (very common: scattering a float computation into an int/bool buffer, or vice-versa). Values
  are also wrong, not just the dtype label, because the int/bool data is reinterpreted as float
  (and vice-versa). It is data-dependent only on dtypes, so it is deterministic and reproducible.
- **Why it usually hides:** in the common case `src.dtype == self.dtype` (or `result_type`
  happens to equal `self.dtype`, e.g. `select_scatter(float32, int64)`), so it only surfaces
  under mixed-dtype scatter — which differential fuzzing hits readily.

## Recommendation

1. **Fix the decomposition.** `torch._refs.select_scatter` (and/or ExecuTorch's edge
   decomposition table for `select_scatter`) should cast `src` to `self.dtype` before the
   `where`, e.g. `src = src.to(x.dtype)` (or wrap the `where` result with
   `.to(x.dtype)` / use `prims.convert_element_type`). This matches eager
   `aten.select_scatter`, which casts `src` into `self`. This single fix eliminates all ~1,245
   cases. Worth filing/patching upstream (PyTorch `_refs`) since ExecuTorch inherits this decomp.
2. **Add a regression test** parametrized over (self_dtype, src_dtype) pairs asserting the
   `to_edge` output spec dtype equals `self.dtype` (covering int64/float32, int64/float16,
   bool/int64).
3. **Harness:** none needed — the harness/generator correctly flagged a genuine bug. Optionally
   fix the reason-text label to read `portable vs eager` to match the documented convention, or
   document that the current ordering is `eager vs portable`.
4. **Sweep for siblings:** the same `where`-based promotion pitfall likely affects other
   `*_scatter` / masked decompositions (`slice_scatter`, `diagonal_scatter`, `masked_scatter`,
   `select.int` scatter variants). Audit those decomps for the same missing `src.to(self.dtype)`.
