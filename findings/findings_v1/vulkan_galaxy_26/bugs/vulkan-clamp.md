# clamp divergence is almost entirely inherited; tensor-bound clamp is not even on the GPU

**Signature:** root op `clamp` — `delta` (~291) and `nonfinite` (~44)
**Cluster size:** 335 born-here (2nd biggest born-here cluster). **318 / 335 are `inherited: true`**; only 17 are `inherited: false`. Flagged-op carriers: `clamp` (17), `var` (11), `min` (7), `split_copy` (7), `erf` (7), `logical_or` (7), `logit` (7), `t_copy` (7), `asin` (7), `unbind_copy` (6), `unfold_copy`, …
**Classification:** **mostly `int64-limitation` / `fp16-precision` (inherited)**, with the residual `inherited:false` cases being a **localizer mis-attribution artifact** (op-name aliasing), not a clamp kernel bug. No real born-here bug found in the Vulkan clamp kernel itself.

## What happens

Example jobs (eager confirmed on host; device value from localizer `detail`):

| job | clamp node | min / max | input dtype | eager `e[0]` | device `b[0]` | inherited |
|-----|-----------|-----------|-------------|--------------|---------------|-----------|
| `w19:360` | n13 `clamp.Tensor_out` | min=None, max=tensor(2,1,1) broadcast | n1 fp32 (from int64 `_to_copy` chain) | -0.3601 | -1.158 | true |
| `w33:553` | n15 `clamp.Tensor_out` | min=None, max=tensor([-0.94, 1.17]) | n10 fp32 | -0.9419 | -0.4825 | true |
| `w63:502` | n13 `clamp.Tensor_out` | min=tensor([0.68,-1.29]), max=None | n12 fp16 (unbind of fp16 tan) | 0.6802 | 0 | true |
| `w4:251`  | n2  `clamp.out`        | min=None, max=0.0 | n0 **int64** (pixel_unshuffle of int64) | -3 | -inf | true |
| `w32:529` | n2  `clamp.Tensor_out` | min=None, max=tensor([0.32,0.65]) | n0 fp32 (bitwise_and of int64) | -7 | nan | true |
| `w0:271`  | n14 `clamp.Tensor_out` | min=tensor([2.06,0.14,0.08]), max=None | n5 fp16 | 5 | nan | **false** |

## Root cause

There are **two distinct mechanisms**, and neither is a bug in the clamp kernel:

### 1. Tensor-bounded clamp never runs on the GPU (so it can't be born-here)

Vulkan registers **only `aten.clamp.default`** — `pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/impl/UnaryOp.cpp:165`:
```cpp
VK_REGISTER_OP(aten.clamp.default, clamp);   // the ONLY clamp registered
```
and `op_registry.py:239` lists only `exir_ops.edge.aten.clamp.default`. The `clamp.Tensor` / `clamp.Tensor_out` variants (tensor-valued min/max, used by w19/w33/w63/w32/w0/w0:448/w1:249/…) are **not registered and not decomposed** (no rewrite in `backends/vulkan/_passes/` or partitioner). The partitioner therefore leaves them on the **portable CPU** kernel. A CPU kernel computes clamp correctly, so the divergence the localizer sees at these nodes is carried in from the node's **input**, which was produced upstream on the GPU. Hence `inherited: true`.

The scalar-bound path that *does* run on the GPU is correct: `add_unary_op_node` / `get_val_or_inf` (UnaryOp.cpp:36-95) map a `None` bound to `±inf`:
```cpp
float get_val_or_inf(ComputeGraph& graph, const ValueRef& val, bool max) {   // :72
  if (!graph.val_is_none(val)) return graph.extract_scalar<float>(val);
  return max ?  std::numeric_limits<float>::infinity()
             : -std::numeric_limits<float>::infinity();
}
```
The GLSL kernel `unary_op.glsl` (clamp variant from `unary_op.yaml`: `OPERATOR: clamp(X, A, B)`) does `t_out[i] = T(clamp(float(t_in[i]), minimum, maximum))` (`runtime/graph/ops/glsl/unary_op.glsl:52`). `clamp(x, a, +inf)` and `clamp(x, -inf, b)` are well-defined; no sentinel/`-1.0` (`kDummyFloat`) leaks into the clamp path. The `-inf`/`nan` device values come from the **input**, not the bound.

### 2. The actual divergence is upstream int64 / fp16 (inherited)

For the inherited cases the first-divergence is genuinely at clamp's input value:
- **w4:251 (`b=-inf`)** and **w32:529 (`b=nan`)**: clamp's input descends from an **int64** tensor (`pixel_unshuffle(int64)`, `bitwise_and` of int64). The Adreno GPU has no 64-bit int; the value was silently truncated/mangled before reaching clamp — see `vulkan-int64-truncation.md`. clamp then clamps the already-broken value.
- **w63:502, w33:553**: clamp's input is fp16 produced by upstream fp16 ops (`tan`/`unbind`, `_softmax`/`t_copy`); the fp16 rounding/saturation difference (see `vulkan-fp16-overflow.md`) shows up first at the clamp node.

### 3. The 17 `inherited:false` cases are localizer name-aliasing, not a kernel bug

`vulkan_localize.py:209` computes `inherited = (root_op != flagged_op)` — it compares **op names**, not node identity. Graphs like `w0:271`, `w11:570`, `w0:448` contain **multiple** clamp nodes (e.g. `w11:570` has `n3=clamp.default(n2,0,0)` *and* `n12=clamp.Tensor_out(...)`; `w0:271` has `n0=clamp.default`, `n14=clamp.Tensor_out`, `n17=clamp.default`). When the originally-flagged user-output is a clamp and the first-divergence node is a *different* clamp, `root_op == flagged_op == "clamp"` makes it look `inherited:false` even though the value entered at a different (often tensor-bound, CPU) clamp whose input came from the GPU. Spot-checking the first elements (e.g. w1:225 `clamp.out(n7=lt-result, -4, None)`: eager `[0,0]`, device `b[0]=0` — actually agree on elem 0; the recorded `e[0]=1` is a different output slot) confirms these are not a real clamp miscomputation.

## Minimal repro

`repro_vulkan-clamp.py` — shows (a) the scalar GLSL clamp path is correct for `min=None`(→-inf)/`max=None`(→+inf) including fp16, reproducing UnaryOp.cpp:72-78 + unary_op.glsl:52 in numpy and matching eager; (b) that `clamp.Tensor_out` is the variant used by every divergent job and is **not** in the Vulkan op set (so it runs on CPU); (c) re-derives the inherited int64 case w4:251 (`-3` → `-inf`) as an upstream int64-truncation feeding clamp, not a clamp bug.

## Notes

- **Inherited**, not a clamp kernel bug. Root causes live in the int64 and fp16 findings:
  - `findings/vulkan_galaxy_26/bugs/vulkan-int64-truncation.md` (w4:251, w32:529 and the int64-fed clamps)
  - `findings/vulkan_galaxy_26/bugs/vulkan-fp16-overflow.md` (w63:502, w33:553 and fp16-fed clamps)
- **hardtanh** shares the exact same unary clamp path (`DEFINE_CLAMP_FN(hardtanh)`, UnaryOp.cpp:153; registered `aten.hardtanh.default` only). Its 42 born-here (38 nonfinite) are the **same story**: 24 are `inherited:true`; the `inherited:false` ones (`hardtanh.out` on an **int64** input, e.g. `e[0]=nan b[0]=-7`) are again the int64/fp16 upstream feeding hardtanh, plus the same op-name-aliasing artifact. No separate hardtanh kernel bug.
- Vulkan-specific surface (only `clamp.default`/`hardtanh.default` registered); portable/xnnpack handle the tensor-bound variants natively so they don't show this partition split.
- **Actionable kernel gap (quality, not the divergence):** `clamp.Tensor`/`clamp.Tensor_out` have no Vulkan kernel; they force a CPU round-trip at the partition boundary. Adding a `clamp.Tensor` builder (per-element min/max via the binary/where path) would close the gap, but it would **not** change any of these results since the current fallback is already correct.
