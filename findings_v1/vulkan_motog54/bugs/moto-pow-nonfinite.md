# Moto G54 (Mali-G57): `pow` drops NaN/Inf — returns a finite value where every other GPU doesn't

**Signature:** born-here root op `pow` — `non-finite (nan/inf positions differ)`
**Cluster:** 81 Moto-only `pow` mismatches (67 non-finite). On the on-device localizer sample,
`pow` is the **#1 born-here non-finite root** (8 of the sampled born-here roots).
**Classification:** **device-specific (budget Mali)** — the same `.pte` produces the correct
NaN/Inf on galaxy/asus/xiaomi/pixel9; only the Mali-G57 diverges. It's the budget GPU's `pow`
implementation, not a delegate bug (it's the loud-tier version of the documented Vulkan `pow`
special-value gap, [../../vulkan_galaxy_26/bugs/vulkan-fp16-overflow.md](../../vulkan_galaxy_26/bugs/vulkan-fp16-overflow.md)).

## What happens (corrected by on-device localization)

Direction matters: the Moto's `pow` returns a **finite** value where eager (and the four other
modern GPUs) produce **NaN/Inf** — it *drops* the special value. First-divergence localizer detail
(eager `e` vs Moto device `b`), all born-here at `pow`:

| job | eager | Moto device | reading |
|---|---|---|---|
| `w0:1416` | `inf` | `6.55e+04` | overflow **saturates to fp16-max (65504)** instead of +Inf |
| `w95:495` | `nan` | `0.4451` | neg-base/frac-exp returns a **finite number** instead of NaN |
| `w12:690` | `nan` | `0.4452` | same |
| `w48:158` | `nan` | `0.09619` | same |
| `w40:559` | `nan` | `0.6875` | same |
| `w66:903` | `nan` | `0` | same (→0) |

## Mechanism

Vulkan computes `pow(x,y)` as `exp(y·log(x))` (or the GLSL builtin). On the Mali-G57:
- **Overflow** (`pow` result > 65504) is **clamped to the fp16 max** (`6.55e4`) rather than going to
  +Inf — the mediump/fp16 path saturates instead of overflowing.
- **Negative base / fractional exponent** (mathematically NaN) returns a **finite** value — the
  budget driver's `pow` lacks the `x<0 → NaN` guard (it effectively computes `exp(y·log|x|)` or
  `pow(|x|,y)`), so it produces a real number where IEEE/eager give NaN.

The flagship GPUs (incl. the Pixel 9's Mali-G715) propagate the NaN/Inf correctly here — so this is
a **precision/edge-handling difference of the budget part**, not the delegate.

## Why this is the loud end of the cross-device axis
pixel9 (flagship Mali): ~0 of these. Moto (budget Mali): 67. The delegate's real `pow` bugs are
device-invariant; the Mali-G57 additionally **silently swallows** the special values (worse: a finite
wrong number is harder to detect than a NaN). Practical guard: clamp `pow` inputs / check the base
sign rather than trusting the GPU's `pow` at the edges — the hazard is worst on low-end parts.

## Evidence
On-device first-divergence localizer (Moto attached): `tmp/run_vulkan_motog54/moto_localized.jsonl`
(generator `vulkan_localize.py --tsv moto_only_mismatch.tsv`). 80% of the 1,200 Moto-only mismatches
are *inherited*; `pow` is one of the genuine born-here roots.
