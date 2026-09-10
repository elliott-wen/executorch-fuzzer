# Vulkan (Galaxy, on-device) differential-fuzz — findings

`corpus/vulkan` (**100,032** auto-generated graphs) lowered with the ExecuTorch **Vulkan** delegate
and executed on a real **Samsung Galaxy** phone (Adreno GPU), jobs dispatched from the host broker
over a `rathole` tunnel, each output diffed against eager PyTorch.

## Verdicts

| verdict | count |
|---|---:|
| OK | 13,331 |
| **MISMATCH** | **8,202** |
| **CRASH** | **6,214** |
| SKIP | 72,285 |
| TIMEOUT | 0 |

Of the ~27.7k graphs that executed (non-SKIP), **~52% diverged**. Full breakdown, mismatch-kind
clusters, the localizer root-op table, and crash op-enrichment: [00-distribution.md](00-distribution.md).

## One-line story

The on-device pipeline works end-to-end; an **on-device first-divergence localizer** collapsed ~8k
mismatches into ~20 born-here root ops (83% of flagged outputs are *inherited* downstream symptoms).
The mismatches are dominated by two inherent GPU realities — **no 64-bit int** and **fp16 compute** —
plus a handful of **genuine vulkan kernel bugs** (NaN-dropping activations, trunc-toward-zero
`floor_divide`, NaN-scrubbing softmax, `sign(NaN)`), and one **serious memory-planning bug** that
silently corrupts a correct output. Crashes carry no device signature (native abort, no logcat) and
are mostly degenerate shapes + int64.

## The headline finding → [bugs/vulkan-copy-elision-aliasing.md](bugs/vulkan-copy-elision-aliasing.md)

**A copy op (`clone`/`lift_fresh_copy`) elided to a buffer alias makes the Vulkan delegate's memory
planner reuse a still-live source buffer, silently corrupting an unrelated sibling output.**
Device-confirmed and deterministic: `out[0] = prod(acos(sigmoid(x)))` returns **2.4668** instead of
**0.8486** purely because a *sibling* output uses an elidable copy — swap the copy for a fresh-buffer
`add 0.0` and it's correct. No crash, no NaN — silent wrong data in a graph whose kernels are all
individually correct. This bug was **prompted by the user**, then reproduced on-device here; the
corpus run hit it ~6% of the time but couldn't recognize it → [WHY-CORPUS-MISSED-IT.md](WHY-CORPUS-MISSED-IT.md).

## Genuine vulkan kernel bugs (real-bug)

| finding | what |
|---|---|
| [bugs/vulkan-tanh-nonfinite.md](bugs/vulkan-tanh-nonfinite.md) | `tanh(NaN)=-1`: the shader does `tanh(clamp(x,-15,15))` and GLSL `clamp`/`min`/`max` drop NaN → finite −1. Same min/max-drops-NaN class covers `relu`/`minimum`/`min`/`max` nonfinite (~90 born-here). |
| [bugs/vulkan-floor-divide.md](bugs/vulkan-floor-divide.md) | integer `floor_divide` **truncates toward zero** (`-15//2 → -7`, eager `-8`): GLSL `floor(int/int)` has no negative-quotient correction. Distinct from the portable bug. |
| [bugs/vulkan-logsoftmax-nonfinite.md](bugs/vulkan-logsoftmax-nonfinite.md) | softmax/log_softmax shader **scrubs NaN/Inf** and clamps the denominator → returns finite where eager propagates NaN. |
| [bugs/vulkan-sign-nan.md](bugs/vulkan-sign-nan.md) | `sign(NaN)=NaN` (ATen mandates `0`) — vulkan instance of the portable `sign-nan` bug. |
| [bugs/vulkan-index-put.md](bugs/vulkan-index-put.md) | `index_put` writes the wrong position: a delegated int64 index *constant* is truncated through the int32 round-trip (aliasing-class). |
| [bugs/vulkan-fp16-overflow.md](bugs/vulkan-fp16-overflow.md) | mixed: `pow(0,neg)→0` and `pow(<0,·)→NaN` (real), int-out `sum` accumulate-then-cast (real), `mean→-65504` fp16 saturation (precision). |

## Inherent limitations & non-bugs

| finding | classification |
|---|---|
| [bugs/vulkan-int64-truncation.md](bugs/vulkan-int64-truncation.md) | **int64-limitation** — Adreno has no 64-bit int; backend silently downcasts to int32 (`INT64_MIN→0`, `INT64_MAX→-1`). The biggest mismatch driver (`_to_copy` ~402). Silent low-32-bit wrap is a quality issue. |
| [bugs/vulkan-clamp.md](bugs/vulkan-clamp.md) | **inherited** — the 2nd-biggest cluster (`clamp`/`hardtanh` ~373) is NOT a clamp bug; the wrong value enters via int64/fp16 upstream. The clamp kernel is correct. |
| [bugs/vulkan-broadcast-shape.md](bugs/vulkan-broadcast-shape.md) | **generator-artifact** — degenerate empty/scalar broadcast + eager out-buffer aliasing; vulkan's shape is actually correct. |

## Shared with portable/xnnpack (reuse, no new work)

[bugs/shared-with-portable-xnnpack.md](bugs/shared-with-portable-xnnpack.md): `remainder` (sign-of-divisor),
`bitwise_left/right_shift` (negative-shift UB), `select_scatter` (where-promotion decomp, backend-agnostic),
`pow` int64 overflow, `sum` cast-order. These reproduce existing [../portable/bugs/](../portable/) and
[../xnnpack/bugs/](../xnnpack/) findings.

## SKIPs & CRASHes

- [skip/README.md](skip/README.md) — 72,285 graceful refusals, each with exact `File.cpp:line` +
  predicate. Defines the effective Vulkan op surface (only ~28% of the corpus runs): missing-shader
  (13k), broadcast guards (7k), >4D norms, dynamic scalars, non-constant weights.
- [crash/README.md](crash/README.md) — 6,214 native aborts. **No device signature is recoverable**
  (abort kills the executor; no logcat path to the remote phone). Characterized by op-enrichment:
  `unfold_copy` 2.9×, `replication_pad3d` 2.6×, `split_with_sizes_copy` 2.1×. Mostly degenerate
  shapes + int64; `split_with_sizes_copy` is the one delegated-op crash candidate (needs logcat).

## Method & caveats

- On-device localizer: [../../tmp/run_vulkan_galaxy_26/vulkan_localize.py](../../tmp/run_vulkan_galaxy_26/vulkan_localize.py)
  re-lowers each graph returning all intermediates, runs on the phone, calls
  `gen.divergence.first_divergence`. Fork-isolated lowering (some all-node graphs native-abort the
  host lowering). `inherited` is op-name based (a measurement artifact when an op repeats).
- The read-only `pytorch_ref` checkout (executorch `2759ef1`) may lag the device's deployed build —
  e.g. `aten.sign` isn't registered in the checkout but runs on-device; findings note this where it
  matters and the fix mirrors the portable one.
- **Crash signatures** and **memory-planning attribution** are the two gaps; both are addressable
  with adb logcat during a targeted re-feed and a memory-plan A/B mutation, respectively (see the
  crash and why-missed docs).

All analysis scripts + raw data live in [../../tmp/run_vulkan_galaxy_26/](../../tmp/run_vulkan_galaxy_26/).
