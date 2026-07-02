# Bug: XNNPACK sqrt / rsqrt return ±0 (not NaN/±inf) on the negative / zero domain

> **Classification:** real XNNPACK-delegate **correctness** divergence on the non-finite domain
> (the magnitude error on finite inputs is acceptable fast-math; losing `NaN`/`±inf` is not).
> Reproduces on the x86 host. Portable computes `sqrt`/`rsqrt` correctly.
>
> Split out of the former `xnnpack-activation-nonfinite.md` (Mechanism B). Sibling mechanisms:
> [clamp/min/max NaN-launder](xnnpack-clamp-minmax-nan.md), [exp NaN/overflow](xnnpack-exp-nonfinite.md).
> Buggy model graph: [`graph_xnnpack-sqrt-rsqrt-domain.py`](graph_xnnpack-sqrt-rsqrt-domain.py).

## Headline

XNNPACK computes `sqrt(x) = x · rsqrt(x)` via the **fast reciprocal-square-root approximation**.
For a **genuine negative real** input (e.g. `-3.0`), `sqrt(x<0)` returns **`-0`** instead of `NaN`
(and `rsqrt` of a negative/zero is likewise wrong). The input is a real negative, not a NaN, so this
is a true **domain** divergence — distinct from the clamp/min/max NaN-laundering family. Eager:
`sqrt(x<0)=NaN`, `rsqrt(x<0)=NaN`, `rsqrt(±0)=±inf`.

## Born-here evidence

| op | job_id | node | input (eager) | eager out | xnnpack out |
|----|--------|------|------|------|------|
| sqrt  | `w0:963`   | n0 | `[-0.482, 1.110]` (real neg) | `[nan, 1.054]` | `-0` |
| sqrt  | `w0:1626`  | n5 | `[-3.0]` (real neg)          | `[nan]`        | `-0` |
| sqrt  | `w10:468`  | n8 | `[-0.464, 0.896]` (real neg) | `[nan, 0.947]` | `-0` |
| sqrt  | `w1:1481`  | n4 | — | `nan` | `-0` |
| rsqrt | `w10:1053` | n5 | `[-0, -1, nan, -inf]` | `[-inf, nan, nan, nan]` | diverges |

Inputs are **genuine negative reals** (e.g. `-3.0`), not NaN — a true domain divergence.

Localizer: `PYTHONPATH=/data/jwen929 QNN_SDK_ROOT="" .venv/bin/python -m mobile.gen.divergence --backend=xnnpack <py>`.

## Root cause

XNNPACK computes `sqrt(x) = x · rsqrt(x)` using the fast reciprocal-sqrt approximation
(`_mm_rsqrt_ps`, 12-bit seed) + Newton-Raphson refinement on the x86 host path
(`xnn_f32_vsqrt_ukernel__sse2_rsqrt_*`, also `avx-rsqrt`, `neon-rsqrt`):

- `.../third-party/XNNPACK/src/f32-vsqrt/gen/f32-vsqrt-sse2-rsqrt.c:71-92`
  - line 71-72: a mask special-cases **only `±0`** inputs (flush-to-zero), nothing for negatives.
  - line 75: `vt0 = xnn_rsqrt_f32(vx)` — `rsqrt` of a negative is unspecified.
  - line 92: `vy = xnn_mul_f32(vx, vt7)` = `vx · rsqrt-approx` → for `vx=-3.0` yields `-0` rather
    than `NaN`.

(The `*-scalar-sqrt` / `*-sse2-sqrt` variants that call a true `sqrt` would give NaN, but the
configured fast path is the rsqrt-based kernel.)

### Eager semantics
ATen `sqrt(x<0) = NaN`, `rsqrt(x<0) = NaN`, `rsqrt(±0) = ±inf`.

## Scope

- Per-op non-finite mismatch propensity (xnnpack vs **portable** in parens): `sqrt` 5.7% (0.4%),
  `rsqrt` 4.7%. A 5–14× spike over portable; portable's `sqrt`/`rsqrt` propagate the non-finite
  domain correctly. Part of the 4,808 `non-finite` MISMATCH rows in
  `tmp/run_xnnpack/skip_reasons_xnnpack.tsv`.

## Buggy model graph

[`graph_xnnpack-sqrt-rsqrt-domain.py`](graph_xnnpack-sqrt-rsqrt-domain.py) — the representative
corpus graph `w0:963`. Run it to lower through XNNPACK and print the first born-here divergence
(`sqrt` n0, `eager=[nan,1.054]` vs `xnnpack=[-0,...]`):

```bash
.venv/bin/python findings/xnnpack_x64/bugs/graph_xnnpack-sqrt-rsqrt-domain.py
# expected: Divergence(node='n0', op='sqrt', kind='nonfinite')  eager=[nan,1.054]  xnnpack=[-0,...]
```

Reproduces in-process on the **x86 host** via the localizer (no phone/worker needed). **Not re-run
here** — no xnnpack host client was available this round; the expected line is the documented result.

## Fix / recommendation

For strict correctness, ExecuTorch should map `sqrt`/`rsqrt` to the **true-sqrt** XNNPACK kernels
(`*-scalar-sqrt`, `*-sse2-sqrt`, `aarch64-neon-sqrt`) rather than the rsqrt-approximation variant,
or not delegate these ops when exact non-finite behavior matters. The rsqrt path's special-case mask
(`f32-vsqrt-sse2-rsqrt.c:71-72`) currently handles only `±0`; it would also need to force `NaN` for
negative inputs.
