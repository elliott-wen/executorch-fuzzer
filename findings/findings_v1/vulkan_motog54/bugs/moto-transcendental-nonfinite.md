# Moto G54 (Mali-G57): `logit` / `gelu` and friends drop NaN where other GPUs keep it

**Signature:** born-here root ops `logit`, `gelu` (and tails of `floor_divide`/`fmod`/`div`) —
`non-finite (nan/inf positions differ)`
**Cluster:** within the 1,200 Moto-only mismatches — `logit` 29 (26 non-finite), `gelu` 24
(19 non-finite), plus `floor_divide` 10 / `fmod` 7 / `div` 7 non-finite. On the localizer sample
`logit` appears as a born-here non-finite root.
**Classification:** **device-specific (budget Mali)** — same special-value-dropping family as
[moto-pow-nonfinite.md](moto-pow-nonfinite.md); not a delegate bug (the other four modern GPUs match
eager on these graphs).

## What happens

These ops are special-value-sensitive: `logit(x)=log(x/(1−x))` → NaN for x∉(0,1) / ±Inf at the
ends; `gelu` uses `erf`/`tanh`; `floor_divide`/`fmod`/`div` → NaN/Inf at by-zero. Eager (and
galaxy/asus/xiaomi/pixel9) produce the NaN/Inf; the Moto returns a finite value instead.

On-device first-divergence localizer (born-here at the named op, eager `e` vs Moto `b`):
- `w24:145`, `w11:250` — root `logit`, eager `nan` (Moto diverges on the row's other positions).
- `gelu` born-here also appears as a small-delta root (`w29:716`: eager `2.512` vs Moto `2.513`) —
  i.e. gelu is *also* a precision carrier, not only special-value.

(Example feed rows: `logit` `w11:1621`,`w16:1367`,`w17:884`; `gelu` `w1:1255`,`w1:620`,`w15:1754`.)

## Mechanism

Same root cause as the `pow` case: on the budget Mali-G57 the transcendental path
(`log`/`exp`/`erf`) is evaluated in looser/lower precision and **without the NaN/Inf guards** that
the flagship parts apply, so:
- `logit` of an out-of-domain input → a finite number instead of NaN (the `log` of a non-positive
  argument doesn't propagate NaN), and
- `gelu`/`erf`/`tanh` saturate at the extremes instead of producing/propagating the reference value.

It is the same "drop the special value, return finite" behaviour as `pow`, across the transcendental
op family — the loud end of the [cross-device numeric axis](../../vulkan_cross_device/README.md).

## Note
These are the same *kind* of divergence the flagship Adreno phones show only narrowly on `sqrt`/
`rsqrt` ([../../vulkan_asus/bugs/asus-transcendental-nonfinite.md](../../vulkan_asus/bugs/asus-transcendental-nonfinite.md));
the Mali-G57 just exhibits it across far more ops (`pow`/`logit`/`gelu`/`div`/`fmod`/`floor_divide`).
Guard: clamp inputs to the valid domain rather than relying on the GPU's special-value handling.
