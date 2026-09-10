# Vulkan delegate: floor_divide(x, x) returns 0 instead of 1

**Signature:** `aten.floor_divide(x, x)` yields `0` for many `x`, where the correct answer is `1`.
**Classification:** real-bug — **kernel** (division rounding + floor), not aliasing.
**Status:** **device-confirmed on the Moto G54 5G** (Mali-G57), in isolation, 2026-06-29.
**Device variance:** Moto-unique in the corpus run — the Pixel 9 computed these graphs correctly.

## Phone-verified behavior

`floor_divide(x, x)` must be `1` for every `x != 0`. Isolated single-op job on the Moto:

```
x      = [0.3, 0.7, 1.1, 1.7, 2.3, 2.9, 3.3, 3.7]
eager  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
device = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]   <-- 0 at x = 1.1, 1.7, 2.9, 3.7
```

Reproduces for positive and negative `x` (`fd_self_mixed`, `fd_self_randn`, `fd_self_neg` all
diverge on the phone). `floor_divide(a, b)` with **distinct** operands (`[5,7,9,11,13,15] // 2`)
is **correct** on the phone (`[2,3,4,5,6,7]`).

The division itself is NOT the cause — verified on the same device:

```
div(x, x)  device = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   <-- exact 1.0, correct
```

So `div(x,x)` returns exactly `1.0` on the phone, yet `floor_divide(x,x)` returns `0` for the same
`x`. The bug is **inside the `floor_divide` lowering**, not in floating-point division. The exact
faulty step in the Vulkan `floor_divide` kernel is not pinned here; only the phone input→output
pairs above are asserted.

## Repro
```
python mobile/findings_v2/vulkan_moto/bugs/repro_vulkan-floor-divide-self.py
```
Needs a connected Vulkan worker (broker on `127.0.0.1:15554`).

## Notes
- Vulkan-specific (portable/eager correct).
- Single-op isolation rules out the memory-planning / aliasing class — this is a true kernel bug.
