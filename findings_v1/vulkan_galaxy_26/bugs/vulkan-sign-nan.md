# Vulkan `sign(NaN)` returns NaN; eager returns 0

**Signature:** root op `sign` — `nonfinite` (eager `0` vs vulkan `nan`)
**Cluster size:** ~20 born-here (localizer), flagged-op carriers (the consumer that surfaced the diff): `unfold_copy` (`w34:418`), `min` (`w19:549`, `w36:1059`). All carry `inherited:true` — the NaN enters `sign` from upstream, and `sign` is the first op whose *value* diverges from eager.

**Classification:** `real-bug` (vulkan computes `sign(NaN)=NaN`, a value a correct ATen-conformant kernel would not — ATen mandates `sign(NaN)=0`). The NaN inputs are themselves generator-artifacts (upstream out-of-domain ops produce NaN on *both* backends), but the divergence at `sign` is a genuine vulkan-vs-ATen semantic bug, identical in kind to the portable bug `findings/portable/bugs/sign-nan.md`.

## What happens

eager: `0`  vs  vulkan (device): `nan`   — example job `w34:418` (localizer `e[0]=0 b[0]=nan`)

| job | flagged op | NaN source upstream | eager `sign` | vulkan `sign` |
|-----|-----------|---------------------|--------------|----------------|
| `w34:418` | `unfold_copy` | `acosh` of a `bool` 0 (domain [1,∞)) → NaN | `0` | `nan` |
| `w19:549` | `min`        | `relu` then `acosh` of values in [0,1) → NaN | `0` | `nan` |
| `w36:1059`| `min`        | `var` with dof≤0 → NaN, `remainder` → NaN     | `0` | `nan` |

Host-confirmed eager (see repro): `torch.sign(nan)=0`, `torch.sgn(nan)=0` for real dtypes.
ATen special-cases NaN → `0`, so eager `sign` *sanitizes* NaN to a finite value; the vulkan
delegate propagates the NaN, then it fans out through the consumer (`min`/`unfold_copy`)
into the compared outputs.

## Root cause

The vulkan kernel computes `sign` arithmetically and does **not** special-case NaN to 0.
PyTorch/ATen `sign` is defined with `sign(NaN)=0` (real signum), so any non-NaN-aware GLSL
formulation diverges on the NaN lane:

- GLSL built-in `sign(x)` returns `0` for `x==0` but is **NaN-propagating** for NaN input
  (the SPIR-V `FSign`/relaxed-float semantics return NaN, not 0), and an arithmetic form
  `(x>0)-(x<0)` would give `0` for NaN — so the observed `nan` means the device build lowers
  `sign` to the NaN-propagating GLSL `sign()` (or `x/abs(x)`-style), not the comparison form.
- The unary-op dispatch path that this would plug into:
  `pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/impl/UnaryOp.cpp:37`
  (`add_unary_op_node`, `DEFINE_ACTIVATION_FN` at the macro block) and the shader
  `pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/glsl/unary_op.glsl:64/76`
  (`t_out[i] = T(op(in_val, ...))`) with the per-op `OPERATOR` from `unary_op.yaml`.

  Note: this read-only reference checkout (executorch HEAD `2759ef1`) does **not** register
  `aten.sign` in `op_registry.py` and has **no `sign` entry in `unary_op.yaml`** — i.e. in
  *this* tree `sign` would fall back to portable CPU. The on-device build that produced the
  localizer values is a newer/different ExecuTorch revision that *does* delegate `sign` to
  vulkan; its `sign` GLSL operator is NaN-propagating. The mechanism is pinned to the unary
  op path above; the exact `OPERATOR: sign(X)` line is in the device build's `unary_op.yaml`,
  not present here. The defect is identical to the portable kernel's documented one
  (`findings/portable/bugs/sign-nan.md`): the kernel fails to map NaN → 0.

Correct fix (mirrors the portable fix): guard NaN to 0 before the signum, e.g.
`OPERATOR: isnan(X) ? 0.0 : sign(X)` (and treat `±inf → ±1`, which GLSL `sign()` already does).

## Minimal repro

`repro_vulkan-sign-nan.py` — shows eager `torch.sign`/`torch.sgn` map NaN→0 (and the exact
`w34:418` / `w19:549` / `w36:1059` upstream chains produce NaN), then shows the GLSL-`sign()`
semantics (NaN-propagating) that the device exhibits, contrasted with the ATen-correct
`isnan?0:sign` form.

## Notes

Same signature as portable: `findings/portable/bugs/sign-nan.md` (eager `sign(NaN)=0`,
backend propagates NaN). This is the **vulkan** instance of the same ATen-conformance gap, on
a different kernel (GLSL `sign()` vs the portable C++ `op_sign.cpp`). Related nan-laundering
family: the portable kernel *returns the NaN* (bug 1), the vulkan softmax *scrubs* NaN to a
finite value (`findings/vulkan_galaxy_26/bugs/vulkan-logsoftmax-nonfinite.md`) — opposite
direction, same root theme of non-ATen-conformant special-value handling. Upstream NaN
feeders (`acosh`<1, `var` dof≤0, `remainder` of NaN) are generator-artifacts shared by both
backends; `sign` is the born-here root and a high-fan-out amplifier.
