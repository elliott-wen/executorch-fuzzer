# QNN graph-optimization bug — `fill.Scalar` output-buffer aliasing

**Failure mode:** mismatch  ·  **Root cause:** graph-optimization (memory-planning / buffer aliasing)
**Device:** QNN HTP x86 emulator (qualcomm, fp16) — device-verified by constant-output probe

## What it is
`torch.ops.aten.fill.Scalar(x, C)` returns a tensor of x's shape filled with the **constant C** —
its output value is independent of x's data and of everything else in the graph. On QNN, when a
graph returns `{fill, <some other output>}`, the device returns **foreign, non-constant values** for
the fill output instead of C. Because fill's correct output is a known constant that cannot depend
on a data-independent sibling, the only possible cause is that **the memory planner aliased the
fill's output buffer with a live tensor**, so the constant was overwritten by the sibling's
computation. This is a buffer-aliasing / memory-planning graph-optimization bug — not a kernel bug.

## Why this is rigorous (and why it works on QNN despite the partitioning limits)
The normal graph-opt confirmation (target correct **alone**, diverges with a sibling) fails on QNN:
a lone `fill.Scalar` won't partition to QNN (`build_job` → SKIP), so "correct alone" can't be shown
by a device run. **The constant-fill probe replaces the device-isolation oracle with the op's
semantics:** fill *must* be all-C, full stop. So a non-C device output in the full graph is
sufficient proof of buffer corruption — no isolation required. The sibling is also verified to be
**data-independent** (not in fill's input cone), ruling out value propagation.

## Device-verified evidence (stable 3/3 each)
| job | target | fill should be | device returns | sibling in fill cone? |
|---|---|---|---|---|
| w100:66 | out[4] `fill.Scalar(n0, 7)` | `7.0` | `[-0.482, 1.11]` | no |
| w102:448 | out[4] `fill.Scalar(n1, -2)` | `-2.0` | `[0.934]` | no |
| w103:514 | out[1] `fill.Scalar(n2, 8)` | `8.0` | `[0.0, 0.015, 0.021]` | no |

The returned values look like another tensor's contents (the aliased buffer), and are identical
run-to-run (not flakiness). `fill.Scalar` is the **#1 GRAPHOPT target (520 cases)** in the bisection —
this is a large class, not a one-off.

## Where the clobber data comes from (pinpointed)
Matching the device's corrupted fill output byte-for-byte against every tensor in the graph
(`tmp/pinpoint_clobber.py`) names the buffer the planner aliased onto fill's output:

| job | fill should be | device returns | **clobber source (100% match)** |
|---|---|---|---|
| w100:66 | `7` | `[-0.4824, 1.1104]` | **`L0` — the input leaf itself** (fill output aliased to the input buffer; the constant write is dropped and the raw input is returned) |
| w102:448 | `-2` | `[0.9341]` | **the `select_copy → alias_copy → constant_pad` view/alias chain** (n0/n1/n2 all share that buffer at value `0.9341`) |
| w103:514 | `8` | `[0.0148, 0.0205, …]` | a small/near-zero buffer not matching any named node — an internal/reused scratch or `out=` buffer |

So the mechanism is concrete: **`fill.Scalar`'s output buffer is aliased with a still-live buffer**
(an input leaf, or a view/alias chain, or a reused scratch buffer); the planner drops/over-writes
fill's constant store, and the device returns whatever already lives in that aliased buffer. The
co-returned sibling is what makes that buffer "live" across the fill, triggering the reuse.

## Reproduce
```
source tmp/qnn_env.sh
BISECT_BACKEND=qualcomm BISECT_CORPUS=corpus_v2/qnn BROKER_PORT=15564 \
  python tmp/qnn_fill_probe.py w100:66 4 3 5
```
Prints, per run, the eager fill constant vs the device's fill output (with a data-independence check
on the sibling). Stable `eager=[C]` vs `device=[non-C…]` confirms the buffer-aliasing bug. Needs the
QNN emulator clients on broker 127.0.0.1:15564.

## It is NOT fill-specific — one general output-buffer-aliasing bug
The clobber-pinpoint across other GRAPHOPT targets shows the **same** mechanism (op's output buffer
aliased onto a still-live tensor; the op's write is dropped) on multiple ops:

| target | device returns | clobber source (100% match) |
|---|---|---|
| `fill.Scalar` | non-constant | input leaf `L0` / view-alias chain |
| `copy` | un-copied value | the `_to_copy → view_copy` chain (returns `self`, copy write lost) |
| `masked_fill.Scalar` | all-`5` | a `broadcast_to(5)` buffer (mask never applied) |

So the real bug is **output-buffer aliasing in QNN memory planning**, surfacing on any op whose
output buffer the planner reuses for a live tensor (`fill` is just the cleanest probe because its
correct answer is a constant). The co-returned sibling is what keeps the rival buffer live across
the op, triggering the reuse.

**`min.unary_out` / `replication_pad1d.out` — chased down, SAME bug (compute-corruption variant).**
These matched no buffer because the corrupted buffer feeds a *computation*, so the wrong output
isn't a verbatim buffer copy. A sibling-sweep (`tmp/sibling_sweep.py`) is decisive for `min.unary_out`
(w100:389): min is **correct with 3 of 4 siblings** (n3/n6/n7 → 0.477) and wrong **only** when the
`fill.Scalar(n4,-1)` sibling (n5) is co-returned (→ 0.0). A buggy min *kernel* would be wrong for
every sibling; being correct with 3/4 proves it is **graph-opt, not operator** — the co-returned
fill corrupts min's input/`out=` buffer so it reduces over wrong data. `replication_pad1d` is
consistent (output corrupted, no buffer match) but only one sibling lowered, so it's less conclusive.

**Conclusion:** there is **one** QNN graph-opt bug — memory-planning **buffer aliasing** — with two
surface forms: (1) **output-buffer aliasing** (fill/copy/masked_fill: the op's write is dropped and a
live buffer's contents are returned verbatim), and (2) **input/out-buffer aliasing into a compute**
(min: a co-returned op's buffer overlaps the target's input/out buffer, so the target computes a
wrong value). No separate fusion/layout graph-opt bug was found.

## How much of the 2,936 GRAPHOPT verdicts is this bug? (587-case sweep)
A sibling-sweep over a 587-case sample (`tmp/sweep_batch.py`), **reconciled with the clobber probe**:

| reconciled class | share | what it is |
|---|---:|---|
| **buffer-aliasing graph-opt bug** | **~56%** | the one confirmed bug, two forms (below) |
| &nbsp;&nbsp;· sibling-specific (sweep "GRAPHOPT") | 37% | input/compute-side aliasing (min-style) — OK with some siblings, wrong with others |
| &nbsp;&nbsp;· all-sibling output-aliasing | 19% | fill/copy/masked_fill — wrong with *every* sibling, so a sibling-variation test mislabels it "operator" |
| genuine-operator candidates | ≤9% | wrong with every sibling AND not a known-aliasing op (transcendentals); upper bound — some may be unprobed aliasing |
| flaky (doesn't reproduce) | ~12% | original mismatch was near-tolerance / non-det |
| inconclusive (<2 siblings lower) | ~23% | can't classify; a chunk are likely aliasing too |

**Key caveat (why two methods are needed):** a sibling-*variation* sweep CANNOT see an
output-aliasing bug that triggers with *every* sibling — it reads as "wrong regardless of context" =
operator. The `fill`/`copy` cases (110 of the 164 "OPERATOR"-bucket rows) are exactly this; the
**constant-output / clobber probe** is what proves they're graph-opt. Net: **≥56% of the GRAPHOPT
verdicts are the buffer-aliasing bug**, ≤9% genuine operator, ~12% flaky, ~23% unclassified.

## Method note (generalizes)
On any backend where ops won't isolate (strict partitioning), use a **constant-output op as a
graph-opt probe**: `fill`/`zeros`/`ones`/`full` have a known answer independent of the rest of the
graph, so any deviation when other outputs are present is provably a buffer/memory-planning bug —
no "lower alone" needed. Added to `analysis.md`.
