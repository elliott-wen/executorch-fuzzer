# var.correction ignores `correction=None` default → biased variance — OPERATOR bug

- **Axis A:** MISMATCH · **Axis B:** **OPERATOR** (not graph-opt — see note) · **Mechanism:** SCALE (N−1)/N
- **CoreML-specific:** portable & xnnpack return the correct value; only CoreML is wrong.

## The bug
`torch.var(x, correction=None)` (equivalently `var.correction(x, dim, correction=None)`) uses the
PyTorch default **correction=1 → unbiased variance, ÷(N−1)**. CoreML instead computes the **biased
variance, ÷N** (i.e. it treats `correction=None` as `correction=0`), so the result is off by
exactly **(N−1)/N**.

## Device-verified (minimal, self-contained)
```
var.correction(randn(2), correction=None):  eager=0.07779  portable=0.07779  coreml=0.03888  coreml/eager=0.4998   [(N-1)/N=0.500]
var.correction(randn(4), correction=None):  eager=0.08314  portable=0.08314  coreml=0.06229  coreml/eager=0.7492   [(N-1)/N=0.750]
```
- coreml's `correction=None` output == coreml's `correction=0` output (biased ÷N), and ≠ its
  `correction=1` output (which is correct). So CoreML defaults `None`→0 instead of `None`→1.
- Reproduces at every N with ratio exactly (N−1)/N; deterministic; portable/xnnpack correct.

Repro: `BROKER_PORT=15554 PYTHONPATH=/data/jwen929 .venv/bin/python
findings/coreml_mac2/bugs/repro_var_correction.py`

## Why it showed up as a "graph-opt SCALE" in the corpus (the confound)
28 of 48 corpus SCALE cases are this bug. In the corpus, `var.correction(…, correction=None)` feeds
a downstream op (`div`, `mul`, `select`, `sinh`, …); the bisector's `bisect_cone` bakes var's
**correct** eager value, so the downstream target passes *alone* and fails only with var **live** →
it was labeled **COMPOSITIONAL** (graph-opt along the path). Running the *trigger* (`var.correction`)
**alone** — the workflow's "run the target alone first" check — reveals it mismatches by itself →
it is an **operator bug in var**, propagated as a (N−1)/N scale onto its consumers, **not** a
graph-opt bug. (The other 20 SCALE cases are unrelated — varied factors ×−1/×2/×3/×0.056/… and
triggers — and need separate per-case triage; not this bug.)

## Fix direction
CoreML's `var`/`std` lowering must honor `correction=None` as **correction=1** (unbiased),
matching aten. Same check applies to `std.correction` and any `var_mean.correction` path.

## Impact
High-reach correctness bug: any model computing variance/std with the default `correction`
(the common case) gets values scaled by (N−1)/N on CoreML — e.g. layer/instance-norm-style
normalizations, statistics layers.
