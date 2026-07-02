# Vulkan delegate: copy-elision / memory-planning aliasing corrupts a sibling output

**Signature:** a *correct* sibling output silently gets the wrong value — not attributable to any
single op kernel.
**Classification:** real-bug — delegate **memory planning / copy-elision**, not a kernel.
**Status:** **device-confirmed on the Moto G54 5G** (deterministic), 2026-06-29. Also confirmed on
the Pixel 9 ([../../vulkan_pixel9/bugs/vulkan-copy-elision-aliasing.md](../../vulkan_pixel9/bugs/vulkan-copy-elision-aliasing.md))
and Galaxy/Adreno — **same bug across three devices and two GPU vendors**.

## What happens

In a multi-output graph delegated to Vulkan, a value-preserving copy of `n0`
(`aten.lift_fresh_copy` or `aten.clone`) is elided to a **buffer alias** of `n0`. The delegate's
memory planner still models the copy as independent, so it treats `n0` as **dead after the copy**
and **reuses `n0`'s buffer** — while `n0` is in fact still live (read by an unrelated branch). A
sibling output that depends on `n0`, *and never references the copy at all*, is computed from the
clobbered buffer → silent wrong data (no crash, no NaN).

## Moto G54 measurement (`x = [-0.4824913740158081, 1.1103534698486328]`)

| graph | sibling copy op (`n1`) | eager `out[0]` | **device `out[0]`** |
|---|---|---:|---:|
| `Bug` | `lift_fresh_copy` | `0.8486` | **`2.4668`** ❌ |
| `BugClone` | `clone` | `0.8486` | **`2.4668`** ❌ |
| `Control` | `add 0.0` (fresh buffer) | `0.8486` | `0.8477` ✅ (fp16 rounding) |

`out[0] = prod(acos(sigmoid(x)))` does **not** read the copied tensor `n1` (`out[1] = acosh(n1)`
does). Swapping *only* `n1`'s producer between an **elidable copy** (`clone`/`lift_fresh_copy`)
and a **fresh-buffer op** (`add 0.0`) flips `out[0]` from wrong to right. The only mechanism by
which an unrelated sibling's buffer choice can change `out[0]` is buffer aliasing/reuse in the
delegate's planner — `sigmoid`/`acos`/`prod` are all individually correct (the `Control` proves
it). The corrupted value `2.4668` is **bit-identical to the Pixel 9**, confirming the same
delegate code path.

## Root cause

The copy is elided to an alias of its source, but the planner models the source as dead at the
copy and reuses its storage while still live → the live consumer reads a clobbered buffer.
Device-level A/B is the evidence; the exact pass (`backends/vulkan/_passes/` + ExecuTorch
memory-planning) is not yet pinned. Behaviour is **sensitive to the output set** — which nodes
are returned changes the planning and can move/hide the corruption.

## Why op-level attribution can't catch it
- **Whole-graph** over-blames `clone`/`lift_fresh_copy`/`unfold_copy`/`narrow_copy` (CRASH 22–40%).
- **Per-output** then blames the *innocent clobbered sibling* (`topk`/`clamp.Tensor_out`/`sum`).
- **Isolation** clears the copy ops (correct alone, [../ISOLATION_PROBES.md](../ISOLATION_PROBES.md)).
Only the A/B repro (swap copy ↔ fresh-buffer op) isolates it.

## Repro
```
python mobile/findings_v2/vulkan_moto/bugs/repro_vulkan-copy-elision-aliasing.py
```
Needs a connected Vulkan worker (broker on `127.0.0.1:15554`).

## Notes
- Vulkan-specific (portable/eager correct).
- `index_put` / `scatter.src_out` drop their writes in single-op isolation
  ([vulkan-index-put-drops-writes.md](vulkan-index-put-drops-writes.md),
  [vulkan-scatter-src-drops-writes.md](vulkan-scatter-src-drops-writes.md)) — filed as separate
  kernel bugs, not attributed to this aliasing class.
- Most serious finding: **silent wrong data** in a graph whose kernels are all individually correct.
