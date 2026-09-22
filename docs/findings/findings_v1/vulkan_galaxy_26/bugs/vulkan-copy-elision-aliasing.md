# Vulkan delegate: copy-elision / memory-planning aliasing corrupts a sibling output

**Signature:** a *correct* sibling output silently gets the wrong value — not attributable to any
single op kernel.
**Classification:** **real-bug** (delegate memory planning, not a kernel).
**Status:** device-confirmed on the Galaxy/Adreno phone (deterministic across runs).

## What happens

In a multi-output graph delegated to Vulkan, a value-preserving copy of `n0`
(`aten.lift_fresh_copy` or `aten.clone`) is elided to a **buffer alias** of `n0`. The delegate's
memory planner then treats `n0` as dead after the "copy" and **reuses `n0`'s buffer**, while `n0`
is in fact still live (read by an unrelated branch). A sibling output that depends on `n0` —
*and never references the copy at all* — is computed from the clobbered buffer.

Device measurement (`x = [-0.4824913740158081, 1.1103534698486328]`):

| graph | sibling copy op | eager `out[0]` | **device `out[0]`** |
|---|---|---:|---:|
| `Bug` | `lift_fresh_copy` | `0.8486` | **`2.4668`** ❌ |
| `BugClone` | `clone` | `0.8486` | **`2.4668`** ❌ |
| `Control` | `add 0.0` (fresh buffer) | `0.8486` | `0.8486` ✅ |

`out[0] = prod(acos(sigmoid(x)))` does **not** read the copied tensor `n1`. Swapping *only*
`n1`'s producer between an **elidable copy** (`clone`/`lift_fresh_copy`) and a **fresh-buffer op**
(`add 0.0`) flips `out[0]` from wrong to right. The only mechanism that lets an unrelated
sibling's buffer choice change `out[0]` is buffer aliasing/reuse in the delegate's planner — the
individual kernels (`sigmoid`/`acos`/`prod`) are all correct (the `Control` proves it).

## Root cause

The copy is elided to an alias of its source, but the planner still models the source as dead at
the copy and reuses its storage while it is still live → the live consumer reads a clobbered
buffer. `clone` and `lift_fresh_copy` behave identically, so it is the **copy-elision class**, not
one op. **Behaviour is proven at the device level; the exact pass/line in the Vulkan delegate's
memory planner / copy-elision (`backends/vulkan/_passes/` + the ExecuTorch memory-planning pass)
is not yet pinned** — the device A/B is the evidence.

The corruption is also **sensitive to the output set**: adding/removing which nodes are returned
changes the planning and can move or hide the corruption (see
`tmp/run_vulkan_galaxy_26/why_missed.py`). That fragility is why a per-node value-diff localizer
cannot attribute it (see [../WHY-CORPUS-MISSED-IT.md](../WHY-CORPUS-MISSED-IT.md)).

## Minimal repro

`repro_vulkan-copy-elision-aliasing.py` (= `tmp/run_vulkan_galaxy_26/check_copy_alias.py`) — lowers
`Bug`/`BugClone`/`Control` to Vulkan, runs all three on the phone via the broker, prints the table
above. Needs a connected Vulkan worker (broker on `127.0.0.1:15554`).

## Notes

- Vulkan-specific (portable/eager both correct).
- Corroborated by the corpus run: `clone`/`alias_copy`/`lift_fresh_copy` are over-represented in
  CRASH graphs (~1.7× baseline, [../00-distribution.md](../00-distribution.md)), and the
  `index_put` "writes/reads the wrong buffer" divergence ([vulkan-index-put.md](vulkan-index-put.md))
  is plausibly the same aliasing class.
- This is the most serious finding in the run: a **silent wrong-data** bug (no crash, no NaN) in a
  graph whose kernels are all individually correct.
