# Vulkan `_log_softmax` launders NaN/Inf into finite garbage

**Signature:** root op `_log_softmax` — `nonfinite` (eager NaN vs vulkan finite)
**Cluster size:** ~14 born-here (localizer), flagged-op carriers: `_log_softmax` (all `inherited:false`, i.e. log_softmax is itself the first divergence)
**Classification:** `real-bug` (vulkan computes a finite value where a correct reduction would propagate NaN; the GLSL shader actively *scrubs* nan/inf and clamps the denominator). The NaN inputs themselves are generator-artifacts (upstream OOB ops), but the divergence at `_log_softmax` is a genuine vulkan-vs-ATen semantic bug.

## What happens

When a row fed to `_log_softmax` contains a NaN (or all-`-inf`), eager propagates NaN
across the whole reduced row; the vulkan delegate instead returns a *finite* number:

| job | eager `out[0]` | vulkan (device) `out[0]` | NaN source upstream |
|-----|----------------|---------------------------|---------------------|
| `w14:1512` | `nan` | `0`      | `rsqrt` of a negative (`mm` row → split → rsqrt) → NaN; reduce dim 0 |
| `w22:1096` | `nan` | `85.19`  | `acos` of `>1` → NaN, then `sub` → NaN; reduce dim 0, size-2 |
| `w14:138`  | `nan` | `nan`    | `log10` of a negative → NaN; reduce dim 0, size-2 |

Eager (host-confirmed, see repro): every job's `_log_softmax` row is all-`nan`. Device
gives `0` / `85.19` / `nan` depending on which shader variant (texture vs buffer, packed
vs non-packed) runs and how Adreno's `max()` treats NaN — i.e. the result is **non-finite
mismatch in 2 of 3 jobs and inconsistent across storage layouts**, which is the tell-tale
of an active nan-laundering path rather than honest NaN propagation.

## Root cause

The vulkan log_softmax shares `softmax.glsl` with `OPERATOR1 = X`, `OPERATOR2 = X - log(Y)`
(so output `= (x - rowmax) - log(sum(exp(x - rowmax)))`), selected in
`pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/glsl/softmax.yaml` (the
`log_softmax_texture3d` variant) and named in
`pytorch_ref/.../graph/ops/impl/Softmax.cpp:124-125` (`kernel_name = "log_" + kernel_name`).

Three independent mechanisms in the shader convert NaN into a finite (wrong) value:

1. **Explicit nan/inf scrub (texture non-packed path)** —
   `pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/glsl/softmax.glsl:96-104`:
   ```glsl
   uvec4 bits = floatBitsToUint(outtex);
   uvec4 nan_inf_mask = uvec4(
       ((bits.x & 0x7F800000u) == 0x7F800000u) ? 0xFFFFFFFFu : 0u, ...);
   outtex = uintBitsToFloat(bits & ~nan_inf_mask);   // zeroes every nan/inf lane
   ```
   Any lane whose exponent field is all-ones (NaN *or* ±Inf) is forced to `0`. This is
   why `w14:1512` (reduce dim orthogonal to packed dim → non-packed texture path) returns
   exactly `0` where eager has `nan`.

2. **Denominator clamp** — `softmax.glsl:94` (`const vec4 safe_denom = max(denominators, vec4(1e-37));`)
   and `softmax.glsl:177` (`const float safe_denominator = max(denominator, 1e-37);`),
   mirrored in `softmax_buffer.glsl:106` (`sum_val = max(sum_val, T(1e-37));`). The comment
   there ("Clamp denominator to avoid 0/0 = NaN when all exp values underflow") states the
   intent — but it also masks the all-`-inf` / NaN-row case, turning `log(0)`/`log(nan)`
   into a finite term and feeding `X - log(Y)` a clean denominator.

3. **`max()` over NaN is implementation-defined** — the row-max reductions
   `softmax.glsl:62/69/134/147` and `softmax_buffer.glsl:79/87` use GLSL `max(a,b)`. With a
   NaN operand Adreno may keep the *finite* operand (per the SPIR-V `FMax`/relaxed rules),
   so `rowmax` comes out finite, `exp(x - rowmax)` is finite for the non-NaN lanes, and the
   `85.19` (job `w22:1096`, size-2 buffer/packed variant) is the resulting
   `(x - rowmax) - log(safe_denom)` garbage. (`w14:138` happens to keep the NaN — same op,
   different layout, different answer: the inconsistency is itself diagnostic.)

A correct log_softmax must propagate NaN: `x - log(sum(exp(...)))` with any NaN in the row
is NaN for every output element (matching ATen / eager). The vulkan shader instead sanitizes
intermediate non-finites, so it disagrees with eager exactly on the non-finite path.

## Minimal repro

`repro_vulkan-logsoftmax-nonfinite.py` — builds the size-2 row `[6.22, nan]` (as in
`w22:1096`) and the rsqrt-of-negative row (as in `w14:1512`), shows eager log_softmax is
all-`nan`, then reproduces the shader's scrub-and-clamp arithmetic in numpy to show it
yields a *finite* result (`0` / a finite number) — matching the recorded device values.

## Notes

vulkan-specific shader bug (the nan/inf scrub at `softmax.glsl:96-104` + denom clamp has no
ATen equivalent). Same *family* as the portable/xnnpack "nan-launder" findings
(`findings/portable/bugs/sign-nan.md`, `findings/portable/bugs/nonfinite-inherited-reposition.md`,
`findings/xnnpack-arm/bugs/graph_neon-gelu-nan-launder.py`) — a kernel sanitizes NaN that
eager would propagate — but here the laundering is in the softmax reduction, not a unary op.
The upstream NaN feeders (`rsqrt`<0, `acos`>1, `log10`<0) are generator-artifacts shared by
both backends; the `_log_softmax` divergence is the born-here root.
