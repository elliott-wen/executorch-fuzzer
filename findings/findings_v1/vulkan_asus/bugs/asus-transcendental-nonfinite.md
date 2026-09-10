# ASUS-specific: transcendental / sqrt special-value non-finites differ from Galaxy

**Signature:** root op `sqrt` / `rsqrt` / `exp` / `sinh` / … — `non-finite (nan/inf positions differ)`
**Cluster size:** 704 ASUS-only mismatches (ASUS wrong, Galaxy = OK); 83% non-finite. `rsqrt`(142) +
`sqrt`(138) = 280.
**Classification:** **device/driver-specific** (not a delegate bug) — two Adreno GPUs evaluate the
same SPIR-V transcendental/sqrt instructions differently at special inputs.

## What happens

On the **same** `.pte`, ASUS places NaN/Inf at different positions than eager PyTorch, while the
Galaxy run matches eager. Confirmed device-specific by the job-by-job comparison
([../00-distribution.md](../00-distribution.md)): these 704 jobs are `MISMATCH` on ASUS and `OK`
(eager-matching) on Galaxy.

Examples (ASUS `MISMATCH out[k] non-finite`, Galaxy `OK`):

| job | flagged op | feeding chain (the special input) |
|---|---|---|
| `w0:1875` | `rsqrt` | `rsqrt(slice(bitwise_or))` — rsqrt of negative/zero |
| `w1:1106` | `rsqrt` | `rsqrt(erf(cumsum))` — `erf ∈ [-1,1]` → rsqrt of a negative |
| `w10:1262` | `sqrt` | `sqrt(min(gelu))` — sqrt of a possibly-negative min |

The common factor across the cluster: **`sqrt`/`rsqrt`/`exp`/`sinh`/`cosh`/`expm1`/`reciprocal`/
`tan`/`log` applied to negative, zero, or large-magnitude inputs** — exactly the inputs where the
IEEE result is NaN/±Inf and where a hardware transcendental unit's approximation / flush behaviour
is implementation-defined.

## Root cause

Both phones run the **identical** Vulkan SPIR-V for these ops (same delegate, same shaders); the
divergence is in the **GPU/driver evaluation of the special-value cases**. Adreno parts differ in:
- how `sqrt`/`inversesqrt` handle negative / zero / denormal inputs (NaN vs flushed-to-0/Inf),
- fast-math / `RelaxedPrecision` (mediump) flushing of out-of-range transcendental results,
- driver version differences in the built-in math libraries.

So eager (`sqrt(neg)=NaN`, `rsqrt(0)=Inf`) is matched by one driver (Galaxy) but not the other
(ASUS), which puts the NaN/Inf at a different position (or returns a finite value instead). This is
the GPU analogue of the [NEON-vs-SSE `f32-vlog` special-value finding](../../xnnpack-arm/bugs/neon-vlog-special-values.md):
there, two CPU ISAs' microkernels disagreed on `log` special values; here, two GPU drivers disagree
on `sqrt`/`rsqrt`/transcendental special values — one layer down, same root pattern.

## Caveat / how to pin exactly

The bulk feed records only "nan/inf positions differ", not the per-position values, so the exact
**direction** (ASUS finite-where-eager-NaN, or ASUS-NaN-where-eager-finite) is not stored. To pin
it, replay a few example jobs on **both** phones and dump values:
```
python feed.py w1:1106 --corpus corpus/vulkan        # with ASUS attached, then with Galaxy attached
```
(or the device localizer detail). The structure already shows it's the sqrt/rsqrt-of-negative class.

## Notes

- Not a correctness bug in the delegate — it's hardware/driver special-value behaviour. The
  delegate's actual bugs are device-invariant (all shared with [Galaxy](../../vulkan_galaxy_26/README.md)).
- Practical impact: a model with `rsqrt`/`sqrt`/transcendentals over data that can reach the
  special-value domain (normalization with tiny/zero variance, `erf`/`gelu` chains, etc.) may give
  NaN/Inf on some Adreno phones and finite values on others — a portability hazard worth a guard
  (clamp inputs to the valid domain) rather than relying on the GPU's edge behaviour.
