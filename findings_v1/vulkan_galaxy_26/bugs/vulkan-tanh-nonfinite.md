# vulkan tanh drops NaN / saturates to a finite -1

**Signature:** root op `tanh` — `nonfinite` (NaN/large-magnitude input collapses to a finite ±1)
**Cluster size:** ~21 born-here (localizer, `kind=nonfinite`, `inherited=false`, all `e[0]=nan b[0]=-1`), flagged-op carriers: `w0:1253`, `w2:1902`, `w1:1049`, `w17:926`, `w18:1097`, `w12:1626`, `w22:774`, `w12:1364`, `w15:1775`, `w14:2016`, `w11:1066`, `w20:1313`, `w13:391`, ... (every nonfinite tanh born-here row)
**Classification:** real-bug (vulkan returns a finite -1 where a correct tanh must propagate NaN)

## What happens
example job `w0:1253` (n5 = `tanh.out(n4)`):
- input n4 = `remainder(div(n0,-6), logical_xor(n0,n0))` = `remainder(x, 0)` = **nan** in BOTH eager and device, so divergence is *born at tanh*, not inherited.
- eager: `tanh(nan) = nan`  vs  vulkan (device): `-1`  (localizer `detail: e[0]=nan b[0]=-1`).

A correct tanh(NaN) must be NaN (IEEE-754 NaN propagation; PyTorch CPU/eager does this). Vulkan instead emits a finite -1.

## Root cause
tanh is registered as a plain activation that dispatches to the `tanh` variant of the generic `unary_op` shader:

- `pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/impl/UnaryOp.cpp:151` — `DEFINE_ACTIVATION_FN(tanh);`
- `pytorch_ref/.../impl/UnaryOp.cpp:80-84` — the macro sets the shader/kernel base name to `"tanh"`.
- `pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/glsl/unary_op.yaml:37-38`:
  ```yaml
  - NAME: tanh
    OPERATOR: tanh(clamp(X, -15.0, 15.0))
  ```
  expanded into `#define op(X,A,B) tanh(clamp(X,-15.0,15.0))` and applied per element in
  `pytorch_ref/.../glsl/unary_op.glsl:51` (buffer) / `:64` (texture).

Mechanism: the `clamp(X, -15.0, 15.0)` wrapper (added to avoid the GLSL `tanh()` builtin overflowing for large |X|) is **not NaN-safe**. GLSL/SPIR-V define `clamp(x,lo,hi)` as `min(max(x,lo),hi)`, and `min`/`max` with a NaN operand are *implementation-defined* — on the Adreno GPU they return the non-NaN argument. So `clamp(nan,-15,15)` → a finite bound (`-15` here), and `tanh(-15) ≈ -1`. The NaN is silently swallowed before `tanh()` ever sees it.

(Note the hardcoded `-15.0/15.0` literals in the OPERATOR mean the A/B push-constants are ignored for tanh — `add_unary_op_node` even passes `kDummyFloat`/`-1` for min/max, `UnaryOp.cpp:83`. The clamp is unconditional and cannot be disabled.)

This is GPU-specific: on host CPU `torch.tanh(torch.clamp(nan,-15,15))` still yields `nan` (verified), because CPU `clamp`/`min`/`max` propagate NaN. The wrong value only appears on the device's GLSL `clamp`.

## Minimal repro
`repro_vulkan-tanh-nonfinite.py` — builds the w0:1253 tanh sub-chain (remainder-by-zero → nan), shows eager `tanh(nan)=nan`, and emulates the GLSL `clamp(nan,-15,15)→-15` Adreno path giving `tanh≈-1`, matching the recorded device value.

## Notes
vulkan-specific (GLSL clamp NaN behavior + the saturating wrapper unique to the Vulkan tanh shader). Not shared with portable/xnnpack (those have no -15/15 clamp; their tanh propagates NaN).

Other activations in `unary_op.yaml`:
- `sigmoid` = `1 / (1 + exp(-1 * X))` (yaml:30) has **no clamp**, so `exp(-nan)=nan` → result nan; NaN propagates correctly. Not affected.
- `gelu` = `0.5 * X * (1 + tanh(...))` (yaml:26) uses an *unclamped* inner `tanh`, multiplied by X; a nan input stays nan. Not affected by this clamp bug.
- Only `tanh` (yaml:38) wraps the input in `clamp(...)`, so it is the sole activation that converts a nonfinite/large input into a finite saturated output.
