# Bug: aliased out= resize divergence — portable silently returns the wrong shape

## One-line statement
This is **not a kernel correctness bug** (no portable kernel computes a wrong shape
for the op it runs), but a real **behavioral execution-model divergence** plus a
**silent robustness gap**: ATen out-variants (`topk` / `min.dim` / `max.dim`) resize
their aliased `out=` buffer **in place at runtime**, and because that buffer is read
by downstream nodes the new shape propagates; ExecuTorch portable uses
**statically-planned buffers and never resizes**, so the aliased tensor keeps its
original `broadcast_to(..., [0,0,0,0,0])` shape. Portable then **silently returns a
wrong-shaped tensor with no runtime check** — e.g. eager `(1,)` vs portable
`(0,0,0,0,0)`.

## Mechanism
The differential generator builds each graph node as an eager out-variant call with
a pre-allocated output tensor of a fixed shape, and deliberately manufactures a
degenerate buffer via `broadcast_to(<scalar/1-elem tensor>, [0,0,0,0,0])`, then
hands that buffer as the `values=` / `indices=` / `min=` / `max=` argument of a
*later* multi-output reduction (`topk`, `min.dim`, `max.dim`):

- **In eager PyTorch**, the multi-output op **resizes the aliased out buffer in
  place** to the operator's true output shape (`topk` k=1 → `(1,)`, `min.dim`
  keepdim=False over a vector → `()`). Because the buffer is aliased and read by
  downstream nodes, eager's returned tuple reflects the resized (small) shape.
- **ExecuTorch portable** lowers each node to a functional out-variant with a
  **statically computed, fixed output shape**. It does not model the in-place resize
  of a tensor aliased by a later node, so the aliased tensor keeps its original
  `broadcast_to(..., [0,0,0,0,0])` shape, and downstream nodes see `(0,0,0,0,0)`.

Net effect: **eager returns the small/scalar shape, portable returns the degenerate
all-zeros shape** — and ExecuTorch neither resizes (like eager) nor raises.

## Reproduction
Real corpus job exhibiting the topk variant (Mechanism A):
- `corpus/portable/w0/w0_1246.py` — out[0]:
  - `n1 = to(n0)` → shape `(1,)`
  - `n2 = broadcast_to(n1, [0, 0, 0, 0, 0])` → `(0,0,0,0,0)` (line 32)
  - `n3 = topk.values(L1, 1, -1, False, False, values=n0, indices=n2)` — writes the
    k=1 index into `indices=n2`, **resizing n2 to `(1,)` in eager** (line 33)
  - `n4 = bitwise_right_shift(n2, …)`; `n5 = logit.out(n4, …)` → out[0]
  - Eager (executing `g`'s node order): the topk resize propagates, so out[0] = `(1,)`.
  - Portable: `n2` stays the static `(0,0,0,0,0)` broadcast shape, so out[0] =
    `(0,0,0,0,0)`.

The eager replay must run nodes in graph order (`m.g(*inputs)` does), so the topk
in-place resize propagates to out[0]; confirmed below to yield `(1,)`.

## Replay
Self-contained in-process replay:
[`replay_shape-aliased-out-resize.py`](replay_shape-aliased-out-resize.py). Loads
the stored corpus job `corpus/portable/w0/w0_1246.py`, runs it on the portable
ExecuTorch runtime and against the eager reference, and asserts the **shapes** of
USER_OUTPUT index 0 differ (exits 0 iff they differ). The comparison is on shape,
not value.

```
$ cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_shape-aliased-out-resize.py 2>&1 \
    | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
job w0:1246 out[0] SHAPE portable=(0, 0, 0, 0, 0)  eager=(1,)
REPLAY: bug REPRODUCED (shapes differ)
```

Eager's `topk` resized the aliased `indices=n2` buffer to `(1,)` and that shape
propagated to out[0]; portable kept the static `(0,0,0,0,0)` broadcast shape — the
documented behavioral divergence, reproduced on `w0:1246` (no fallback job needed).

## Scope
Per [mismatch-shape.md](../mismatch-shape.md), the output-shape MISMATCH cluster is
**485 cases, 100% this single mechanism** (static-shape / aliased-`out=` resize
artifact), and **0 real backend shape-inference defects**:
- **289 / 485 (59.6%)** — the divergent returned node is itself a multi-output
  reduction (`topk.values` / `min.dim_min` / `max.dim_max`) whose
  `values=`/`min=`/`max=` argument is a `broadcast_to(..., [0,0,..])` buffer
  (Mechanism A, direct resize target).
- **196 / 485 (40.4%)** — the divergent node is a downstream consumer that inherits
  the divergence (Mechanism B; e.g. `floor_divide`, elementwise out-ops,
  `unfold_copy`).
- **484 / 485 (99.8%)** chains contain a `broadcast_to(..., [0,0,..])` node;
  **476 / 485 (98.1%)** have a portable-side shape with a degenerate `0` dim.

## Classification
- **Class:** **Behavioral execution-model divergence**, not a kernel correctness
  bug. No portable kernel computes a wrong shape for the op it actually executes.
  The divergence is entirely due to eager's reliance on in-place resizing of
  tensors aliased as `out=`/`values=`/`min=` buffers of later ops — an effect the
  ExecuTorch lowering (functional out-variants, statically planned shapes)
  intentionally does not reproduce.
- **Real kernel bugs in this cluster: 0.** Artifact share: **485 / 485 (100%).**
- It will **not fire on a well-formed exported model** (the memory planner sizes the
  out buffers correctly); the fuzzer manufactures it by aliasing a
  `broadcast_to(..., [0,0,0,0,0])` buffer as a later op's `out=`.
- **The one actionable ExecuTorch-side ask:** ExecuTorch **silently** returns a
  tensor whose shape disagrees with eager — it neither resizes nor raises. That is
  the robustness gap.

## Recommendation
1. **Suppress / filter at the generator/harness level.** Stop generating graphs that
   reuse an earlier node's tensor as the `out=`/`values=`/`indices=`/`min=`/`max=`
   buffer of a *later* op and then read that buffer downstream — in particular stop
   feeding `broadcast_to(x, [0,0,...,0])` buffers into `topk.values` / `min.dim_min`
   / `max.dim_max` out-arguments (the source of 295 of the 485 divergences). In the
   differ, mark such cases `SKIP (aliased-out-resize, static-shape model)` instead of
   `MISMATCH`.
2. **The one genuinely actionable ExecuTorch improvement: add a runtime/lowering
   shape assertion.** Do **not** "fix" portable to chase these shapes — matching them
   would require dynamic in-place resize of statically planned buffers, contradicting
   the backend's design. Instead, assert that an op's produced shape equals its
   statically planned `out=` buffer shape (or that an aliased `out=` is not later read
   with a divergent shape), turning the silent divergence into a detectable error.
   This matters as ExecuTorch grows dynamic-shape support.

## Severity & classification (summary)
- **Severity:** none as a correctness regression (won't fire on a well-formed
  exported model); the actionable item is a **robustness gap** (missing
  planned-vs-produced shape assertion), worth documenting, not suppressing silently.

Cross-reference: [mismatch-shape.md](../mismatch-shape.md). Other reproduced jobs
for the same mechanism: `w0:1007` (min.dim, `()` vs `(0,0,0,0,0)`) and `w1:288`
(min.dim, `(1,)` vs `(0,0,0,0,0)`).
