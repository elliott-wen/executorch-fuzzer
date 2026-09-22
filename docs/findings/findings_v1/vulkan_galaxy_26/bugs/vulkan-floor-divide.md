# Vulkan integer floor_divide truncates toward zero instead of flooring

**Signature:** root op `floor_divide` — `delta` (wrong integer rounding on negative quotients)
**Cluster size:** ~81 born-here (localizer). Flagged-op carriers (consumers of the bad value):
`unfold_copy` (w56:1050), `sub` (w16:1546), `exp`, `log`, `mul`, `maximum`, `div`, `tan`,
`min`, `mean`, `topk`, `select_scatter`, `scatter_add`, … — many flagged ops, one root.
**Classification:** **real-bug** (a correct integer floor_divide floors toward −∞; vulkan
computes a different, wrong value).

## What happens
Eager integer `floor_divide` floors toward −∞; vulkan truncates toward zero.

- `w56:1050` — `floor_divide(L2, n0)` int64, `L2[0]=-4`, `n0[0]=9`: eager `-4//9 = -1`,
  vulkan (device) `n1[0] = 0`. (`detail: e[0]=-1 b[0]=0`)
- `w16:1546` — float16 path, `floor_divide(n1, n0)`, `n1[0]=-2.671875`, `n0[0]=0.381592`:
  eager `-8`, vulkan (device) `-7`. (`detail: e[0]=-8 b[0]=-7`)

Other cluster examples confirmed eager on host: `w41:1186` (`e=-1 b=0`),
`w34:1524` (`e=-1 b=0`), `w30:924` (`e=-1 b=0`), `w23:1079` (`e=5 b=0`),
`w26:666` (`e=3 b=1`). All are negative-quotient cases where trunc ≠ floor.

## Root cause
The vulkan floor_divide kernel is the generic binary op with operator literal
**`floor(X / Y)`**:

`pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/glsl/binary_op_buffer.yaml:29-30`
```yaml
    - NAME: binary_floor_divide_buffer
      OPERATOR: floor(X / Y)
```
applied at `binary_op_buffer.glsl:71` / `:86`:
```glsl
t_out[out_bufi] = T(op(t_in[...], t_other[...], T(alpha)));   // op == floor(X / Y)
```
where `T = buffer_scalar_type(DTYPE)` and the kernel is dtype-specialized via
`add_dtype_suffix(kernel_name, graph.dtype_of(in1))`
(`pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/impl/BinaryOp.cpp:80`,
registered `aten.div.Tensor_mode` → `floor_divide` at `BinaryOp.cpp:152`,
`floor_divide` defined `BinaryOp.cpp:134`).

For **integer** DTYPE this is broken two ways:
1. `X / Y` is GLSL *integer* division, which truncates toward zero: `(-4) / 9 == 0`,
   `(-15) / 2 == -7`.
2. `floor(...)` applied to an already-integral value is a no-op.

So the shader produces `trunc(X/Y)`, never the floored result. Eager
`aten.floor_divide` on ints floors toward −∞ (`-4//9 = -1`, `-15//2 = -8`). A correct
integer floor needs the classic correction
`q = X / Y; if ((X % Y != 0) && ((X < 0) != (Y < 0))) q -= 1;` — the shader has no such
term. (`floor(X / Y)` only works if `X / Y` is computed as a *float*; for ints it is not.)

The float16 case (`w16:1546`) is a secondary contributor: in fp16 the quotient
`-2.671875 / 0.381592` rounds to exactly `-7.0`, so `floor(-7.0) = -7` while eager (computing
the true ratio `-7.0019`) floors to `-8`. That one is fp16-rounding-on-the-quotient on top of
the same `floor(X/Y)` formulation; the dominant ~81-cluster mechanism is the integer-trunc bug.

## Minimal repro
`repro_vulkan-floor-divide.py` — shows eager `-4//9=-1`, `-15//2=-8` (floor) vs the GLSL
`floor(X/Y)` integer formulation which yields `0`, `-7` (trunc), matching the device values.

## Notes
Vulkan-specific. The portable finding
[`findings/portable/bugs/floor_divide.md`](../../portable/bugs/floor_divide.md) documents a
*different* bug (portable float ÷0 → ±inf) and explicitly proves portable's **integer**
floor_divide is CORRECT (floors toward −∞). Vulkan's integer path is the bug that portable
doc disproves for portable. Not shared. (See also
`findings/vulkan_galaxy_26/bugs/shared-with-portable-xnnpack.md` "NOT shared" table.)
