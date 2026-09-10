# vulkan fp16 reductions lose NaN/Inf (saturate to ±65504), pow drops special values, int-out sum reduces in float then casts

**Signature:** root op `mean` / `pow` / `sum` — `nonfinite` (mean/pow) and `delta` (sum)
**Cluster size:** `mean` nonfinite ~9 (`w0:1016`, `w56:1345`, `w4:313`), `pow` nonfinite ~70 (`w74:62`, `w57:181`), `sum` delta ~57 (`w1:757`, `w38:1209`)
**Classification:** mixed — `fp16-precision` for the mean ±65504 saturation (Adreno mediump has no IEEE NaN/Inf); `real-bug` for both pow cases (safe-pow swallows `0^neg`; Tensor_Tensor uses raw undefined `pow(<0,·)`) and for the int-out `sum` (vulkan accumulates-then-casts; eager casts-then-accumulates).

## What happens

| job | node | op | eager | vulkan (device) |
|-----|------|----|-------|-----------------|
| `w0:1016`  | n0  | `mean.out` of empty `(0,)`           | `nan`  | `-6.55e+04` (=−65504) |
| `w56:1345` | n3  | `mean.out([nan, 0.46])`              | `nan`  | `-6.55e+04` |
| `w4:313`   | n1  | `mean.out([-0.50, nan])`            | `nan`  | `-6.55e+04` |
| `w74:62`   | n16 | `pow.Tensor_Scalar([0,0], -1)`      | `inf`  | `0` |
| `w57:181`  | n10 | `pow.Tensor_Tensor([-1], [-1,1,1])` | `-1`   | `nan` |
| `w1:757`   | n5  | `sum.IntList(fp16[0.881×4]) -> int64`| `0`   | `3` |
| `w38:1209` | n2  | `sum.IntList(fp16[-0.48,1.11]) -> int64`| `1`| `0` |

(eager values reproduced on host; device values from localizer `detail`.)

## Root cause

### A. mean → ±65504 saturation (fp16-precision)
`add_reduce_node` picks the shader variant by **output** dtype and there is no fp32 accumulator:
- `pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/impl/Reduce.cpp:144` — `add_dtype_suffix(kernel_name, graph.dtype_of(out));`
- `pytorch_ref/.../glsl/reduce.glsl:43` declares `shared vec4 shared_vecs[...]` and `:94/:155` `vec4 accum` — the accumulator dtype follows `VEC4_T = texel_load_type(DTYPE,STORAGE)`, i.e. **half** for the half variant. The input intermediate (n0/n1) is stored in a half texture.
- `pytorch_ref/.../glsl/reduce.yaml:20-21` — `mean` `POSTPROCESS: (accum / tin_sizes[reduce_dim])`. For the empty-tensor case (`w0`) that is `0 / 0`.
- `pytorch_ref/.../gen_vulkan_spv.py:870-880` — for all-half variants `PRECISION` is auto-downgraded `highp → mediump`. On Adreno `mediump float` is fp16 with **no IEEE NaN/Inf**: a NaN/Inf input (or a `0/0`) is flushed to the largest representable magnitude, `±65504`. The observed `-65504` is the clamped −fp16-max — the saturation happens in the **GPU's fp16 ALU**, not in any explicit `clamp` in the shader.

So a NaN that eager produces upstream (asin/acosh of out-of-domain, or mean-of-empty) cannot survive the half reduction: it is read as / collapses to ±65504. Documented magnitude: exactly ±65504 = ±(2−2⁻¹⁰)·2¹⁵.

### B. pow special values (real-bug, two distinct paths)
- `pow.Tensor_Scalar` (`w74` n16) → `pytorch_ref/.../impl/BinaryScalarOp.cpp:72-73` → shader `pow_scalar_*`, `OPERATOR: power_of(X,Y)` (`glsl/binary_scalar_texture.yaml:9`). The "safe" `power_of` in `pytorch_ref/.../glsl/binary_op_defs.glslh:23-27`:
  ```glsl
  T power_of(T x, T y) {
    if (x == 0.0) {
      return (y == 0.0) ? T(1.0) : T(0.0);   // <-- 0^(negative) returns 0, must be +Inf
    }
  ```
  `pow(0, -1)` → returns `0`; eager → `inf`. The `x==0` short-circuit ignores the sign of `y`.
- `pow.Tensor_Tensor` (`w57` n10) → `pytorch_ref/.../impl/BinaryOp.cpp:138` → shader `binary_pow_*`, `OPERATOR: pow(X, Y)` (`glsl/binary_op_texture.yaml:28-29`) — the **raw GLSL builtin**, which the comment in `binary_op_defs.glslh:14` itself flags as *"undefined for x < 0"*. The Tensor_Tensor variant never calls the safe `power_of`, so `pow(-1, -1)` is undefined → Adreno yields `nan`; eager → `-1`. (Asymmetry: the scalar path was hardened, the tensor-tensor path was not.)

### C. int-out sum: accumulate-then-cast vs cast-then-accumulate (real-bug)
For `sum` with a floating input and an **int64** out and `dtype=None`, eager casts each element to the output int dtype **before** summing (`torch.sum(x, dtype=int64)` semantics): `w38` `int(-0.48)+int(1.11)=0+1=1`; `w1` `int(0.881)×4=0`. Vulkan's reduce instead accumulates in the **float** texel and casts only the final result on store (`reduce.glsl:101` `accum + new_val`, write at `:132`): `w38` `0.628→0`; `w1` `3.53→3`. Both device values match the accumulate-then-cast order, opposite of eager. (int64 itself is not GPU-native, but the divergence is the cast-order, not the 64-bit range — values are tiny.)

## Minimal repro
`repro_vulkan-fp16-overflow.py` — for each of the three mechanisms: shows the eager value, then emulates the vulkan path (fp16 saturation for mean, the `power_of`/raw-`pow` shader funcs for pow, accumulate-then-cast for int-out sum) reproducing the recorded device value; references the source file:line in comments.

## Notes
Vulkan-specific in all three cases:
- (A) the ±65504 saturation is the Adreno mediump fp16 ALU + half-only accumulator (`reduce.glsl` has no fp32 accum, unlike portable/xnnpack which accumulate in fp32 and propagate NaN). Same family as `vulkan-tanh-nonfinite.md` (mediump/Adreno loses nonfinite) but here via the half accumulator, not a clamp.
- (B) the `power_of` 0^neg bug and the raw-`pow` Tensor_Tensor undefined-domain are unique to the Vulkan GLSL shaders.
- (C) the cast-order is unique to the texel-based reduce; a correct impl would cast inputs to the out dtype before accumulating when `dtype` is unset and out is integral.
