# ARM bug: NEON `tan` / `erf` reduced-precision (value drift)

**Class:** ARM-NEON precision divergence (fast-math) · **Verdict on phone:** MISMATCH (value
delta) · **Scope:** ~37 `tan` + ~32 `erf` ARM-only rows (x86 == eager)

## Buggy model graph (run to trigger on the phone)
```bash
.venv/bin/python findings/xnnpack-arm/bugs/graph_neon-tan-erf-precision.py
# or:
python feed.py w12:1473 --corpus corpus/xnnpack --host 127.0.0.1   # tan
python feed.py w10:1362 --corpus corpus/xnnpack --host 127.0.0.1   # erf
```
**Expected:** `status MISMATCH  max|delta|≈0.30` (tan, `w12:1473` out[3]) / `≈0.84` (erf,
`w10:1362` out[7]) under `rtol=0.01 atol=0.001`. (Prereq: broker + ARM phone connected.)

## What happens
ARM's NEON `tan`/`erf` microkernels use a **lower-degree / lower-precision polynomial** than the
scalar/x86 path, so finite outputs drift by **more than the strict `atol=0.001`**. Unlike the
[`f32-vlog`](neon-vlog-special-values.md) and [`gelu`](neon-gelu-nan-launder.md) bugs, this is a
**value-accuracy** divergence (no Inf/NaN involved) — the results are "slightly wrong", not
catastrophic. On x86 the same jobs match eager (localizer "no divergence"), so it is the NEON
approximation specifically.

## Root cause (where to look)
NEON `tan`/`erf` microkernels under
`pytorch_ref/executorch/backends/xnnpack/third-party/XNNPACK/src/f32-vtan/` and `.../f32-verf/`
(NEON variants), vs the scalar reference. XNNPACK's NEON transcendentals trade accuracy for speed;
the corpus's strict tolerance surfaces the gap.

## Classification & fix
**Borderline** — arguably acceptable fast-math, but it *does* exceed eager beyond the harness
tolerance and is ARM-only. Two reasonable responses: (1) scale the differential-fuzz tolerance by
output dtype / op class so benign NEON precision doesn't flag (recommended for triage), and/or
(2) use a higher-precision NEON variant (or scalar fallback) for `tan`/`erf` when bit-accuracy
matters. Lower severity than the special-value bugs.
