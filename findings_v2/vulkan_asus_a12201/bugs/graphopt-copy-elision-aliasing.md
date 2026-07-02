# Vulkan graph-optimization bug — copy-elision / aliasing (ASUS a12201)

**Failure mode:** mismatch  ·  **Root cause:** graph-optimization (buffer planning / copy-elision)
**Device:** ASUS a12201 (Vulkan) — device-verified by A/B test (`tmp/confirm_graphopt.py`)

## What it is
`expand_copy` (and `clone`) compute the **correct** result when returned as the sole graph output,
but their result is **corrupted when another (sibling) output is co-returned** in the same graph.
The co-returned sibling changes ExecuTorch's memory planning so the copy/view op's output buffer is
aliased/elided and overwritten — a graph-level optimization bug, not a kernel bug (the op is correct
in isolation).

## Device-verified evidence (A/B, on the ASUS)
For each case: **A)** graph returning the target alone → builds READY and compares **OK**;
**B)** graph returning target + the required sibling → **MISMATCH** on the *target's* output.

| target | example job | A) alone | B) +sibling | instances confirmed |
|---|---|---|---|---|
| `expand_copy` | w111:349 out0 (sibling out1) | OK ×(4–5) | MISMATCH ×(4–5) | **4/4** (w111:349, w117:568, w18:223, w21:331) |
| `clone` | w115:215 out0 (sibling out4) | OK ×4 | MISMATCH ×4 | 2/3 (w115:215, w122:117) |

This is the same **copy-elision / aliasing** class confirmed on the Moto Vulkan, now independently
device-confirmed on the ASUS — a backend-level ExecuTorch-Vulkan defect, not device-specific.

## Reproduce
```
BISECT_BACKEND=vulkan BISECT_CORPUS=corpus_v2/vulkan BROKER_PORT=15554 \
  python tmp/confirm_graphopt.py w111:349 0 1 5
```
Prints A) target-alone (×5) and B) target+sibling (×5); a stable `OK` vs `MISMATCH` split confirms
the optimization (not the kernel) is at fault. Needs a connected Vulkan worker on 127.0.0.1:15554.

## NOT graph-opt (ruled out by the same test)
- `index_put` — target alone is a **runtime SKIP** on the device, so "correct alone" can't be
  established; its GRAPHOPT label is an artifact (the bisector reads SKIP-alone as "no reproduction").
- `transpose_copy.int`, `alias_copy`, some `clone` — **flaky**: the full graph does not re-diverge,
  so the original GRAPHOPT verdict was a near-tolerance / non-deterministic bisection.

The raw 903 GRAPHOPT count therefore overstates the real bug; the confirmed graph-opt finding is the
**copy-elision/aliasing class** above.
