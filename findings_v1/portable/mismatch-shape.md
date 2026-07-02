# MISMATCH cluster: output shape mismatch (485)

> **Per-bug split (with a runnable replay):**
> [bugs/shape-aliased-out-resize.md](bugs/shape-aliased-out-resize.md) — the aliased-`out=`
> in-place-resize divergence + the missing runtime shape assertion. Full index: [REPLAY.md](REPLAY.md).

## Summary

**Headline: this is not a kernel *correctness* bug (no portable kernel computes a wrong shape
for the op it runs), but it IS a real, fuzzer-discovered *behavioral divergence* between eager
and ExecuTorch portable in out-variant resize / aliasing semantics — and a silent-no-runtime-check
robustness gap worth documenting, not suppressing.**

The interesting axis here is the **execution model**, not arithmetic. ATen out-variants resize
their `out=` buffer in place at runtime; because that buffer is aliased and read by downstream
nodes, the new shape propagates. ExecuTorch portable lowers to functional out-variants with
**statically planned buffers and never resizes**, so the alias keeps its original shape. Same
program, two different runtime shape semantics — and ExecuTorch returns the wrong-shaped tensor
**without erroring**. (It will not fire on a well-formed exported model, where the memory planner
sizes buffers correctly; the fuzzer manufactures it by aliasing a `broadcast_to(...,[0,0,0,0,0])`
buffer as a later op's `out=`. So: not a ships-in-production break, but a genuine structural
difference + a missing runtime guard.)

The differential generator builds each graph node as an eager `out=`-variant call with a
**pre-allocated output tensor** of a fixed shape (e.g. `out=torch.empty((0,0,0,0,0), ...)`).
Many graphs deliberately create a degenerate buffer via
`broadcast_to(<scalar/1-elem tensor>, [0,0,0,0,0])` and then hand that buffer as the
`out=` / `values=` / `indices=` / `min=` / `min_indices=` / `max=` argument of a *later*
multi-output reduction (`topk`, `min.dim`, `max.dim`).

- **In eager PyTorch**, those out-variant / multi-output ops **resize the aliased out
  buffer in place** to the operator's true output shape (e.g. `topk` k=1 → `(1,)`,
  `min.dim(keepdim=False)` over a vector → `()`). Because the buffer is aliased and read
  by *downstream* nodes, eager's returned tuple reflects the resized (small) shape.
- **ExecuTorch portable** lowers each node to a functional out-variant with a **statically
  computed, fixed output shape**. It does **not** model the in-place resize of a tensor that
  is aliased by a later node, so the aliased tensor keeps its original
  `broadcast_to(...,[0,0,0,0,0])` shape. Downstream nodes therefore see `(0,0,0,0,0)`.

Net effect: **eager returns the small/scalar shape, portable returns the degenerate
all-zeros shape.** (The results TSV `reason` is formatted `EAGER_shape vs PORTABLE_shape`,
confirmed by reproduction — see below.)

Quantification over the 485 cases:
- **484 / 485 (99.8%)** chains contain a `broadcast_to(..., [0,0,...])` (`bcast`) node.
- **476 / 485 (98.1%)** have a portable-side shape with a degenerate `0` dim.
- **289 / 485 (59.6%)** the divergent returned node is *itself* a multi-output reduction
  (`topk.values` / `min.dim_min` / `max.dim_max`) whose `values=`/`min=`/`max=` argument is
  a `broadcast_to(...,[0,0,..])` buffer (direct resize-target).
- **196 / 485 (40.4%)** the divergent node is a *downstream consumer* that inherits the
  shape divergence (e.g. `floor_divide`, elementwise out-ops, `unfold_copy`) operating on a
  tensor that eager already resized but portable did not.

Real backend shape-inference defects in this cluster: **0 identified.** Static-shape /
aliasing harness artifact: **485 / 485 (100%).**

## Dominant ops (producer of the divergent `out[i]`)

| count | op producing divergent out[i] | role |
|------:|-------------------------------|------|
| 136 | `topk.values` | multi-output; `values=`/`indices=` buffer is a zero-broadcast |
|  96 | `min.dim_min` | multi-output; `min=`/`min_indices=` buffer is a zero-broadcast |
|  63 | `max.dim_max` | multi-output; `max=`/`max_indices=` buffer is a zero-broadcast |
|   6 | `maximum.out` | downstream consumer of resized/aliased tensor |
|   5 | `ne.Scalar_out` | downstream consumer |
|   4 | `sinh.out`, `logical_or.out`, `bitwise_not.out`, `alias_copy.default` | downstream consumers |
|   3 | `floor_divide.default`, `unfold_copy.default`, `prod.int_out`, `log.out`, `sqrt.out`, `sin.out`, `sign.out`, `cosh.out`, `acosh.out`, `clone.default`, `eq.Scalar_out`, `ne.Tensor_out`, `bitwise_and.Scalar_out`, `bitwise_left_shift.Tensor_Scalar`, `remainder.Scalar`, `lift_fresh_copy.default` | downstream consumers |

`topk.values + min.dim_min + max.dim_max = 295 (61%)` of divergent nodes are the
multi-output reductions at the heart of the mechanism; the long tail are elementwise/copy
ops downstream of an aliased tensor that one side resized.

## Mechanisms (each with a reproduced example)

### Mechanism A — multi-output reduction resizes an aliased zero-broadcast `out=` buffer (dominant, ~60%)

The returned divergent node is the reduction itself. Its output buffer is a
`broadcast_to(scalar, [0,0,0,0,0])`. Eager resizes it to the reduction's true shape;
portable keeps the zero shape.

**Example 1 — `w0:1246`, out[0], `topk.values`** (`corpus/portable/w0/w0_1246.py`)
- TSV reason: `out[0] shape (1,) vs (0, 0, 0, 0, 0)` (eager `(1,)` vs portable `(0,0,0,0,0)`)
- Relevant lines:
  - `n1 = _to_copy(n0)`  → shape `(1,)`
  - `n2 = broadcast_to(n1, [0,0,0,0,0])` → eager `(0,0,0,0,0)`
  - `n3 = topk.values(L1, 1, -1, False, False, values=n0, indices=n2)`  ← writes k=1 index into `indices=n2`, **resizing n2 to `(1,)`**
  - `n4 = bitwise_right_shift(n2, 8)`; `n5 = logit.out(n4)`  (out[0])
- Pre-allocated out for n5: `torch.empty((0,0,0,0,0), dtype=float16)`.
- Reproduced shapes for the chain (eager, executing g's node order):
  ```
  after topk n2= (1,)          # topk resized the aliased indices buffer
  n4= (1,)                     # bitwise_right_shift sees the resized n2
  n5= (1,)                     # eager out[0]
  ```
  Portable (full run): `out[0] = (0,0,0,0,0)` (n2 stays the static broadcast shape).
- Proof eager only diverges because of node ordering: if n4/n5 are computed *before* topk,
  eager n5 is also `(0,0,0,0,0)`; computed *after* topk (g's actual order) it is `(1,)`.
  Portable never sees the resize regardless of order.

**Example 2 — `w0:1007`, out[2], `min.dim_min`** (`corpus/portable/w0/w0_1007.py`)
- TSV reason: `out[2] shape () vs (0, 0, 0, 0, 0)`
- `n15 = broadcast_to(n14, [0,0,0,0,0])`; `n16 = min.dim_min(L2, 0, False, min=n15, min_indices=n12)` (out[2]).
- Eager: `min.dim` resizes `min=n15` to the reduced shape `()`; portable keeps `(0,0,0,0,0)`.
- Reproduced: portable `out[2]=(0,0,0,0,0)`, eager `out[2]=()`.

**Example 3 — `w1:288`, out[2], `min.dim_min`** (`corpus/portable/w1/w1_288.py`)
- TSV reason: `out[2] shape (1,) vs (0, 0, 0, 0, 0)`
- `n10 = broadcast_to(n5, [0,0,0,0,0])`; `n11 = min.dim_min(L1, -1, True, min=n10, min_indices=n4)` (out[2]).
- Eager resizes `min=n10` to `(1,)` (keepdim=True, reduce last dim of a vector); portable keeps `(0,0,0,0,0)`.
- Reproduced full run:
  ```
  portable: ... out[2]=(0,0,0,0,0) out[3]=(0,0,0,0,0) ...
  eager:    ... out[2]=(1,)        out[3]=(1,)        ...
  ```

### Mechanism B — downstream consumer inherits the aliased-resize divergence (~40%)

The divergent node is an ordinary elementwise/copy/reduction op, but one of its inputs is a
tensor that a *prior* multi-output op resized in eager (and didn't in portable). Often the
final shape is a broadcast of the two sides, so the two backends produce *different*
degenerate shapes (e.g. `(0,)` vs `(0,0,0,0,0)`).

**Example 4 — `w0:1096`, out[5]→`floor_divide`** (`corpus/portable/w0/w0_1096.py`)
- TSV reason for the cluster entry: `out[4] shape () vs (0, 0, 0, 0, 0)` (min.dim case);
  the floor_divide divergence is the out[5] entry `(0,) vs (0,0,0,0,0)`.
- `n13 = broadcast_to(n7, [0,0,0,0,0])`  (n7 is a `()` scalar from `argmax`)
- `n14 = min.dim_min(L2, -1, False, min=n13, min_indices=n10)`  ← **resizes n13 → `()` in eager**
- `n15 = slice_copy(n6, 0, 0, 0, 1)` → `(0,)`
- `n16 = floor_divide(n15, n13)`  (out[5])
- Reproduced:
  ```
  n13 after min.dim = ()      (eager)   -> floor_divide((0,), ())   = (0,)
  n13 stays (0,0,0,0,0) (portable)      -> floor_divide((0,),(0,0,0,0,0)) = (0,0,0,0,0)
  ```
  Portable out[5]=`(0,0,0,0,0)`, eager out[5]=`(0,)`. Pure downstream propagation of A.

### Sub-family — non-degenerate aliasing/rank/keepdim differences (9 cases, ~2%)

The same aliased-resize mechanism but the broadcast target is not all-zeros, so the shapes
differ without a `0` dim. Examples:
- `w24:1639 (1,4) vs (1,1,4)`, `w66:1695 (4,2,2) vs (1,4,2,2)`, `w90:992 (3,1,2) vs (1,3,1,2)`
  — rank/keepdim mismatch between the pre-allocated buffer and the reduction's true shape.
- `w39:3010 () vs (1,)`, `w87:380 () vs (1,)` — scalar vs 1-element from a reduction buffer.
- `w42:707 (1,1) vs (1,1,1,1,1)`, `w44:161 (2,2,2,2) vs (1,1,1,1)`, `w44:1527 (1,1) vs (1,)`,
  `w73:1288 (4,) vs (4,1)` — analogous fixed-rank vs broadcast/keepdim divergence.

Same root cause (static pre-allocated shape not following eager's in-place resize); the only
difference is the absence of a literal `0` dim.

## Classification & severity

| Mechanism | count | share | type | severity |
|-----------|------:|------:|------|----------|
| A: multi-output reduction resizes aliased zero-broadcast `out=` buffer | 289 | 59.6% | harness/static-shape artifact | none (not a backend bug) |
| B: downstream consumer inherits the aliased-resize divergence | 187 | 38.6% | harness/static-shape artifact | none |
| Sub-family: non-degenerate rank/keepdim/scalar aliasing divergence | 9 | 1.9% | harness/static-shape artifact | none |
| **Real shape-inference / upper-bound defect** | **0** | **0%** | — | — |

**Artifact vs real-bug split: 485 / 485 (100%) artifact, 0 / 485 real backend bug.**

Severity: **not a kernel correctness bug, but a documented behavioral divergence + a real
robustness gap.** No ExecuTorch portable kernel computes a wrong output shape for the op it
actually executes. The divergence arises because the eager reference relies on **in-place
resizing of tensors that are aliased as `out=`/`values=`/`min=` buffers of later ops** — an
effect the ExecuTorch lowering (functional out-variants, statically planned shapes)
intentionally does not reproduce.

**Why it's still an interesting discovery (and not just noise):**
- It is a genuine, empirically-surfaced difference in **runtime out-variant / alias semantics**
  between the two backends — discovered from data, not docs.
- ExecuTorch **silently** returns the wrong-shaped tensor: it neither resizes (like eager) nor
  raises. When the static-shape assumption is violated, there is **no runtime check** — that's
  the robustness gap.
- It probes the exact boundary that matters for **dynamic / data-dependent shapes**, where the
  static-buffer assumption is most likely to be stressed.

It will not trigger on a well-formed exported model (the memory planner sizes the out buffers
correctly), so it is not a production correctness regression — but it is the kind of structural
seam a differential fuzzer is uniquely good at exposing.

## Recommendation

1. **Suppress this cluster at the generator/harness level.** The mismatches are a modeling
   gap in the *reference*, not the backend. Specifically:
   - Stop generating graphs that reuse an earlier node's tensor as the `out=` / `values=` /
     `indices=` / `min=` / `min_indices=` / `max=` / `max_indices=` buffer of a *later*
     op, **and** then read that buffer downstream. This is the aliasing that eager resizes
     and ExecuTorch cannot.
   - In particular, stop feeding `broadcast_to(x, [0,0,...,0])` buffers into `topk.values`,
     `min.dim_min`, `max.dim_max` out-arguments (the source of 295 of the 485 divergences).
2. **Filter in the differ.** When the eager reference performs an out-variant/multi-output
   op whose `out=` tensor is also a graph node consumed elsewhere, mark the case
   `SKIP (aliased-out-resize, static-shape model)` instead of `MISMATCH`. A cheap detector:
   any returned tensor whose shape ancestry passes through a `broadcast_to(...,[0,...])`
   that is later passed as an out-kwarg.
3. **The one genuinely actionable ExecuTorch-side improvement: add a runtime/lowering shape
   assertion.** Do *not* "fix" portable to chase these shapes — matching them would require
   dynamic in-place resize of statically planned buffers, which contradicts the backend's
   design. But the real gap this finding exposes is that ExecuTorch **silently** returns a
   tensor whose shape disagrees with eager. A cheap guard — assert that an op's produced shape
   equals its statically planned `out=` buffer shape (or that an aliased `out=` is not later
   read with a divergent shape) — would turn this silent divergence into a detectable error,
   which matters as ExecuTorch grows dynamic-shape support.
4. Optionally keep a tiny watch on the **9 non-degenerate cases** — they are the same
   mechanism, but if any future case shows a rank/keepdim divergence *without* an aliased
   `out=` buffer in its ancestry, that would be a genuine shape-inference bug worth a second
   look. None observed here.

### Reproduced jobs cited
- `w0:1246` (topk, A) — eager `(1,)` / portable `(0,0,0,0,0)`
- `w0:1007` (min.dim, A) — eager `()` / portable `(0,0,0,0,0)`
- `w1:288`  (min.dim, A) — eager `(1,)` / portable `(0,0,0,0,0)`
- `w0:1096` (floor_divide downstream of min.dim, B) — eager `(0,)` / portable `(0,0,0,0,0)`
- Also confirmed portable-side all-zeros for `w0:677, w0:695, w0:860` during reproduction.
