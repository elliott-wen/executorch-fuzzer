# ARM bug: NEON `f32-vlog` drops Inf/NaN (log / logit → finite junk)

**Class:** real ARM-NEON correctness divergence · **Verdict on phone:** MISMATCH (non-finite) ·
**Scope:** ~2,843 ARM-only rows (logit ~2,385 + log ~458; plus floor_divide/layer_norm carriers)

## Buggy model graph (run to trigger on the phone)
```bash
.venv/bin/python findings/xnnpack-arm/bugs/graph_neon-vlog-special-values.py
# or one job directly:
python feed.py w0:1114 --corpus corpus/xnnpack --host 127.0.0.1
```
**Expected:** `out[2] eager=[Infinity, Infinity]  DEVICE(ARM)=[88.376266, 88.376266]`.
(Prereq: `python broker.py` running + the ARM phone connected as a worker.)

## What happens
On the **ARM phone**, `log(+Inf)` returns the finite constant **`88.376266`** (and `log(0)`/
`log(x<0)` return finite junk) instead of `+Inf`/`-Inf`/`NaN`. The **x86 host** XNNPACK agrees
with eager on the same jobs (the localizer reports "no divergence"). `logit(x) = log(x/(1-x))`
is decomposed to this `log`, so the whole logit cluster inherits it.

## Root cause (file:line)
XNNPACK selects the kernel by arch in
`pytorch_ref/executorch/backends/xnnpack/third-party/XNNPACK/src/configs/unary-elementwise-config.c`:
- aarch64 (phone) **:1429-1431** → `xnn_f32_vlog_ukernel__neon_rational_3_3_div_u8`
- x86 **:1432-1466** → AVX/SSE2/**scalar** (`f32-vlog-scalar-log.c:34` = `logf(vx)`, correct)

The NEON kernel `.../src/f32-vlog/gen/f32-vlog-neon-rational-3-3-div.c` has **no special-value
handling** — it bit-extracts the exponent (`xnn_signed_getexp_f32`), forces the mantissa into
`[1,2)`, and evaluates a degree-3/3 rational polynomial unconditionally (`vy = vexp*ln2 + poly`).
For `x=+Inf` the exponent field is max (255) → `xnn_signed_getexp_f32` yields a finite ~128 and
the mantissa is masked into `[1,2)` → `vy ≈ 128·ln2 + residual ≈ 88.4` (the observed
`88.376266`/`88.781731`). No `x<=0`/`x==Inf`/`isnan` branch, unlike the scalar `logf`.

## Evidence (x86 == eager; phone shows ARM value)
| job | chain → log input | eager | ARM device |
|---|---|---|---|
| `w0:1114` | `cosh→sinh`→`+Inf`, `logit` | `[Inf, Inf]` | `[88.376266, 88.376266]` |
| `w0:1419` | softmax/compare → `logit` | `[NaN, NaN]` | `[88.781731, 88.781731]` |
| `w0:292` | `prod`→`+Inf`, `logit` | `[Inf]` | `[88.376266]` |
| `w0:1602` | `sign`→`±1`, `logit(±1)=±Inf` | `[Inf, Inf]` | `[88.376266, 88.376266]` |

Each job's x86 localizer (`python -m mobile.gen.divergence --backend=xnnpack corpus/xnnpack/w0/w0_1114.py`)
reports `no divergence`. **Carriers:** `floor_divide`/`native_layer_norm` non-finite rows
(~584/~409) are downstream of a NEON transcendental/fp16-overflow non-finite that ARM never
produces (NEON log/sigmoid saturate finite) or launders to 0 in the normalize/divide step — same
root family, different output positions.

## Fix
Add Inf/NaN/x≤0 guards to `f32-vlog/rational-3-3.c.in` (mirror `logf`: `x<0→NaN`, `x==0→-Inf`,
`x==+Inf→+Inf`), or fall back to the scalar kernel on non-finite inputs. Until then, don't lower
`log`/`log2`/`log10`/`logit` (and layer-norm reciprocal paths) to XNNPACK on ARM for models that
can see overflow/out-of-domain inputs.
