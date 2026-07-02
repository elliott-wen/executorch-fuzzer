# ARM bug: NEON `gelu` launders NaN → 0

**Class:** real ARM-NEON correctness divergence · **Verdict on phone:** MISMATCH (non-finite) ·
**Scope:** ~37 ARM-only `gelu` mismatch rows (x86 == eager)

## Buggy model graph (run to trigger on the phone)
```bash
.venv/bin/python findings/xnnpack-arm/bugs/graph_neon-gelu-nan-launder.py
# or:
python feed.py w0:1273 --corpus corpus/xnnpack --host 127.0.0.1
```
**Expected:** a `float16` output shows `eager=[1.657227, nan]  DEVICE(ARM)=[1.657227, 0.0]` —
ARM turned the `nan` lane into `0.0`. (Prereq: broker + ARM phone connected.)

## What happens
ARM's `gelu` NEON path **flushes a `NaN` input lane to `0.0`** where eager (and x86) keep `NaN`.
Confirmed on `w0:1273`: `out[4]` eager `[1.657227, nan]` vs ARM `[1.657227, 0.0]` (other lanes,
e.g. out[5]/out[6], agree — only the laundered position differs, which is why the feeder reports
"non-finite positions differ").

## Why it's distinct from the shared activation cluster
The host-x86 run already documented a **shared** relu/clamp/min/max NaN→0 laundering (both arches
use SIMD `fmin`/`fmax` that return the non-NaN operand — see
[../../xnnpack/bugs/xnnpack-activation-nonfinite.md](../../xnnpack/bugs/xnnpack-activation-nonfinite.md)).
This one is **ARM-only**: on x86 `gelu` keeps the `NaN` (localizer "no divergence"), so it's the
NEON `gelu` microkernel specifically (gelu = `x·0.5·(1+erf(x/√2))`; the NEON erf/clamp path drops
the NaN that the scalar/SSE path preserves). It is **not** the `f32-vlog` magic-constant pattern —
the value is a clean `0.0`, not finite junk.

## Root cause (where to look)
gelu lowers to XNNPACK's gelu/erf NEON microkernel under
`pytorch_ref/executorch/backends/xnnpack/third-party/XNNPACK/src/f32-vgelu/` (NEON variant) — the
NaN is dropped by the clamp/`fmin`/`fmax` step inside the gelu evaluation, mirroring the shared
activation issue but on a path that x86 happens to keep finite-correct. (Not yet pinned to an
exact line — the replay confirms the behavior; the kernel read is the remaining step.)

## Fix
Make the NEON gelu/erf path NaN-propagating (don't use bare `fmin`/`fmax` clamps on NaN), or fall
back to scalar on non-finite inputs — same remedy family as the activation NaN-laundering.
