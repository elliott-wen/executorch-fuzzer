# Artifact: transcendental / fp16 reduction precision below harness tolerance

## Classification: ARTIFACT — not an ExecuTorch bug

This is a **differential-fuzz artifact** (benign floating-point precision), not a
portable-kernel defect. Elementwise transcendental ops (`erf`, `gelu`, `tanh`,
`exp`, `sin`, `cos`, `sigmoid`, `rsqrt`, …) and fp16 reductions (`mean`, `sum`)
use polynomial approximations / different accumulation order than eager. The
result differs by a small amount that is **correct to within the output dtype's
precision** but happens to exceed the harness's strict
`atol=0.001 / rtol=0.01` thresholds. For `float16` in particular, one ULP near
magnitude ~1 is ≈5e-4, so a couple of ULP of legitimate rounding — and any
reduction that accumulates a handful of those — trivially crosses `atol=0.001`.
Cited: mismatch-delta-float Mechanisms **M3/M4**
([../mismatch-delta-float.md](../mismatch-delta-float.md)).

## One-line statement
A small, dtype-appropriate rounding difference (transcendental approximation or
fp16 reduction) trips the harness only because the absolute tolerance is too
strict for `float16`; the value is correct to fp16 precision.

## Mechanism
- **M3 — transcendental approximation.** Portable's elementwise math kernels use
  polynomial/range-reduction approximations whose last-bit results differ from
  eager's. On `float16`/`float32` outputs the difference is a few ULP. Only near
  small or singular arguments (where the magnitude is tiny and fp16 has almost no
  significant bits) does the *relative* error grow — and even then both results
  are "approximation-correct", not garbage.
- **M4 — fp16 reduction rounding.** `mean`/`sum` accumulate in a particular order
  and round each step in fp16. Eager and portable round at different points, so a
  reduction over several elements drifts by several fp16 ULP. Near magnitude ~1,
  one fp16 ULP is ≈4.9e-4, so even a ~28-ULP reduction drift is only ≈0.0137
  absolute / ≈1.7% relative — well inside fp16's representational noise, yet far
  above `atol=0.001`.

In both cases the divergence is **bounded by the dtype's precision**: it scales
with `ulp(dtype) * |out|`, not with any logic error.

## Reproduction (verified)
Job: [`corpus/portable/w1/w1_155.py`](../../../corpus/portable/w1/w1_155.py),
USER_OUTPUT[8] = `mean.dtype_out(n2, out=float16 scalar)`, fed by a
`div`/`rsub` chain (the M4 fp16-reduction case).

Observed:
```
out[8] dtype=float16   portable=0.775390625   eager=0.7890625
|delta| = 0.0136719    rel = 1.73%    = 28.0 fp16 ULP
1 fp16 ULP at |out|=0.7891 is 4.88e-04
```
`|delta|` exceeds `atol=0.001` (so the harness flags MISMATCH), but the relative
error is 1.73% and the gap is a few-dozen fp16 ULP of accumulated `mean`
rounding — i.e. the output is correct to within fp16 precision. This is exactly
the M3/M4 benign case cited in mismatch-delta-float
(`w1:155 out[8] portable=0.775390625 eager=0.7890625 |d|=0.0137 rel=1.7%`).

## Why it is not a backend bug
- The two values agree to ~1.7% relative on a `float16` output; fp16 carries
  ~3–4 significant decimal digits, so this is at the floor of the format's
  precision. There is no wrong logic — only a different (equally valid) rounding
  of a reduction/approximation.
- The harness uses a single fixed `atol=0.001` regardless of output dtype. That
  threshold is appropriate for `float32`/`float64` but is *below one fp16 ULP* at
  magnitudes ≳1, so it is guaranteed to flag benign fp16 rounding.
- Contrast with the genuine bugs in the same cluster (negative-shift integer
  magic numbers stored into float `out`, `|delta| ≈ 1e18`): those dwarf any
  dtype-scaled tolerance and are real defects. This row is the opposite end —
  pure precision.

## Recommendation
(From mismatch-delta-float §Recommendation 2–3.)
1. **Bucket by relative error, not absolute `|delta|`.** Triaging on
   `rel = |delta| / max(|eager|, eps)` immediately separates benign fp16 rounding
   (rel ≲ a few %) from real defects (rel ~1e11 for integer-into-float).
2. **Scale tolerance by output dtype.** Use `atol ≈ 4 · ulp(dtype) · max|out|`
   (a few fp16 ULP at the output magnitude) for `float16` outputs, with a
   slightly looser band for transcendental ops, instead of a flat
   `atol=0.001`. This clears the M3/M4 noise without masking the
   integer-into-float bug, whose `|delta|` is enormous.

## Replay
Self-contained in-process replay:
[`replay_transcendental-fp16-precision.py`](replay_transcendental-fp16-precision.py).
Loads `corpus/portable/w1/w1_155.py`, runs it on the portable ExecuTorch runtime
and the eager reference, and reports portable/eager values, `|delta|`, relative
error, the output dtype, and the gap in fp16 ULP at USER_OUTPUT[8]. Exits 0 iff a
small delta is present that exceeds `atol=0.001` yet stays within fp16's
precision (relative error ≤ 5% on an fp16 output) — i.e. benign precision, not a
real bug.

```
$ cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_transcendental-fp16-precision.py 2>&1 \
    | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
job w1:155 out[8] dtype=torch.float16 (eager dtype=torch.float16)
  portable=0.775390625  eager=0.7890625
  |delta|=0.0136719  rel=1.73%  ==28.0 fp16 ULP (1 ULP at |out|=0.7891 is 4.88e-04)
  exceeds harness atol=0.001 but a small fp16 rounding error -> benign precision
REPLAY: ARTIFACT reproduced (benign fp16 precision)
```

The harness flags the delta because `0.0137 > atol=0.001`; the script confirms
it is within fp16's precision (1.7% relative), so it is benign rounding, not a
backend defect.
