# Bug: XNNPACK exp returns 0 on NaN (and garbage on overflow) — no non-finite guard

> **Classification:** real XNNPACK-delegate **correctness** divergence on the non-finite path.
> Reproduces on the x86 host. Portable computes `exp` correctly.
>
> Split out of the former `xnnpack-activation-nonfinite.md` (Mechanism C). Sibling mechanisms:
> [clamp/min/max NaN-launder](xnnpack-clamp-minmax-nan.md), [sqrt/rsqrt domain](xnnpack-sqrt-rsqrt-domain.md).
> Buggy model graph: [`graph_xnnpack-exp-nonfinite.py`](graph_xnnpack-exp-nonfinite.py).

## Headline

XNNPACK's delegated `exp` is a **rational-polynomial approximation** whose range reduction uses
integer bit-manipulation with **no NaN/Inf guard**, so `exp(NaN) → 0` (and a large-overflow input
produces a garbage finite value). Eager: `exp(NaN)=NaN`, `exp(+large)=+inf`.

## Born-here evidence

| op | job_id | node | input (eager) | eager out | xnnpack out |
|----|--------|------|------|------|------|
| exp | `w0:938`  | n12 | `[nan]` | `nan` | `0` |
| exp | `w0:785`  | n6  | `[nan]` (fp32) | `nan` | `0` |
| exp | `w10:464` | n12 | `[nan]` | `nan` | `0` |
| exp | `w11:1095`| n8  | — | `nan` | `0` |
| exp | `w11:1339`| n6  | — | `nan` | `0` |

Localizer: `PYTHONPATH=/data/jwen929 QNN_SDK_ROOT="" .venv/bin/python -m mobile.gen.divergence --backend=xnnpack <py>`.

## Root cause

The configured host `exp` kernel is the **rational-polynomial** approximation
`xnn_f32_vexp_ukernel__{sse2,avx,fma3,avx512f}_rational_3_2_div_*`
(`.../src/configs/unary-elementwise-config.c:1352-1388`), **not** the libm `expf` scalar kernel.
Its range reduction uses integer bit-manipulation magic with **no NaN/Inf guard**:

- `.../src/f32-vexp/gen/f32-vexp-scalar-rational-3-2-div.c:20-36`
  (`xnn_setexp_f32`: `(x + 8388735.0) << 23`; `xnn_qd_round_f32`: `(x + magic) - magic`)

Feeding `NaN` (or a large-overflow input) through the `(x+magic)<<23` integer path produces a
garbage finite value (observed `0`); there is no `NaN → NaN` / `overflow → +inf` handling.

### Eager semantics
ATen `exp(NaN) = NaN`, `exp(+large) = +inf`.

## Scope

- Per-op non-finite mismatch propensity (xnnpack vs **portable** in parens): `exp` 6.3% (0.5%) — a
  ~13× spike, all introduced by the delegate. Part of the 4,808 `non-finite` MISMATCH rows in
  `tmp/run_xnnpack/skip_reasons_xnnpack.tsv`.

## Buggy model graph

[`graph_xnnpack-exp-nonfinite.py`](graph_xnnpack-exp-nonfinite.py) — the representative corpus graph
`w0:938`. Run it to lower through XNNPACK and print the first born-here divergence (`exp` n12,
`eager=nan` vs `xnnpack=0`):

```bash
.venv/bin/python findings/xnnpack_x64/bugs/graph_xnnpack-exp-nonfinite.py
# expected: Divergence(node='n12', op='exp', kind='nonfinite') e[0]=nan b[0]=0
```

Reproduces in-process on the **x86 host** via the localizer (no phone/worker needed). **Not re-run
here** — no xnnpack host client was available this round; the expected line is the documented result.

## Fix / recommendation

The rational-poly exp kernel needs a NaN/Inf guard (propagate NaN, saturate to `+inf` on overflow),
or ExecuTorch should route `exp` to the libm `scalar-exp` (calls `expf`, which is correct) when exact
non-finite behavior is required. Today the configured host kernel is the approximation with no guard.
A delegate-level "strict IEEE non-finite" mode (refuse to partition the no-guard `exp`) would also
eliminate this cluster.
