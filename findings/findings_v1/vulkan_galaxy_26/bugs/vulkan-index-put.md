# Vulkan index_put writes to the wrong position (int64 index constant lowered to GPU → 0)

**Signature:** root op `index_put` — `delta` (divergence at the position index_put writes to;
the copy-through region is *correct*, but the written/accumulated index is wrong)
**Cluster size:** ~99 born-here (localizer). Flagged-op carriers (consumers): `unfold_copy`
(w76:1620), `rsqrt` (w2:24, w93:968), `sub` (w12:119, w25:539), `sign` (w15:184, w16:1164),
`atanh`, `exp`, `_log_softmax`, `bitwise_or`, `index`, … — one root op, many flagged outputs.
**Classification:** **real-bug** (vulkan lowering corrupts a delegated int64 index constant;
a correct lowering would write to the requested index). Has an int64-limitation flavour (the
index is int64 and vulkan has no native 64-bit int) but the value `1` fits in int32, so this is
a lowering/staging bug, not an unavoidable int64 truncation.

## What happens
`index_put(self, [torch.tensor(1, int64)], value, accumulate)` should leave index 0 unchanged
and write/accumulate at **index 1**. On device the value lands at **index 0** instead — i.e.
the GPU-supplied index is `0`, not `1`. The unwritten region (index 1) is left untouched and so
*matches* eager; the divergence shows up at index 0 (the copy-through value).

example `w76:1620` (accumulate=True): `n0=mul=[-0.5357,-0.4507]`, value `L2=1.1636`.
eager `n1 = [-0.5357, 0.7129]` (index1 += value).  vulkan (device) `n1[0] = 0.6279`.
Note `-0.5357 + 1.1636 = 0.6279` → vulkan **accumulated into index 0**. (`detail: e[0]=-0.5357 b[0]=0.6279`)

Confirmed eager-on-host that "index → 0" reproduces every device value exactly:
| job | acc | eager `[0]` | device `b[0]` | self[0] (+value if acc) |
|---|---|---|---|---|
| w76:1620 | True | -0.5357 | 0.6279 | -0.5357 + 1.1636 = **0.6279** |
| w2:24    | True | -0.4825 | 0.4515 | -0.4825 + 0.9340 = **0.4515** |
| w25:539  | False| -0.4825 | 0.934  | overwritten by value = **0.9340** |
| w15:184 / w16:1164 / w43:1038 / w37:1326 | False | -0.4825 | 0.934 | same value-at-0 |

All match "index value 1 became 0".

## Root cause
index_put is **not** a registered vulkan op (no `IndexPut.cpp`/glsl; not in
`backends/vulkan/op_registry.py`), so it stays on the portable CPU kernel
(`pytorch_ref/executorch/kernels/portable/cpu/op_index_put.cpp`) — and that kernel is correct
(verified: running the same graph through the *portable* ExecuTorch runtime on host gives the
right answer `[-0.4825, 2.0444]`).

The bug is in the **vulkan partitioner / lowering**. The index `torch.tensor(1)` is a lifted
int64 constant (`c_lifted_tensor_0 → lift_fresh_copy → detach_`). The partitioner pulls this
constant-only chain into a delegated GPU subgraph:
```
%lowered_module_0 = get_attr        # delegated subgraph:
%executorch_call_delegate(...)      #   placeholder c_lifted_tensor_0 (int64 scalar = 1)
%getitem = call_delegate[0]         #   -> dim_order_ops._clone_dim_order(...)  (no-op copy)
%aten_index_put_default(x, [%getitem], v, True)   # CPU kernel, index from GPU
```
So the int64 0-d index constant makes a round-trip through the GPU (`_clone_dim_order`) before
reaching the CPU index_put. The Adreno GPU has no native 64-bit int (`shaderInt64` is off,
`runtime/vk_api/Device.cpp:94`), so int64 tensors are staged through int32 with a hard-coded
cast in/out:
`pytorch_ref/executorch/backends/vulkan/runtime/graph/ComputeGraph.cpp:965`
```cpp
if (src_data_dtype == vkapi::kLong && staging_dtype == vkapi::kInt) {
  staging->cast_and_copy_from<int64_t, int32_t>(casted_data, numel);   // copy IN
...
// ComputeGraph.cpp:1008
if (dst_data_dtype == vkapi::kLong && staging_dtype == vkapi::kInt) {
  staging->cast_and_copy_to<int32_t, int64_t>(casted_data, numel);     // copy OUT
```
The value `1` fits in int32, so the cast itself wouldn't lose it — but the GPU round-trip of
this **0-d int64 scalar constant** through the clone/texture path returns `0`, which is what the
device-recorded values prove. The fix is either to not partition a constant-only index-feeding
subgraph onto the GPU, or to correctly round-trip a 0-d int64 scalar; the symptom is an index of
`0` regardless of the true constant.

(For the few `inherited:false` cases such as `w0:1305` `e[0]=9 b[0]=-1`, `w14:1614`
`e[0]=1 b[0]=2`, the index/self is itself int64 produced on-GPU upstream, so int64 corruption
compounds — same family, GPU int64 handling.)

## Minimal repro
`repro_vulkan-index-put.py` — builds `index_put(self,[tensor(1)],v,acc)`; runs eager and the
correct portable ET path (writes index 1), and shows that forcing the index to 0 (what the GPU
delegate returns) reproduces the device values for w76:1620, w2:24, w25:539.

## Notes
Vulkan-specific. No matching portable/xnnpack born-here finding (the portable/xnnpack
index_put entries are all `inherited:true`, divergence flowing in from a prior op, kernel
itself correct). The portable ET runtime computes this graph correctly on host. See
`findings/vulkan_galaxy_26/bugs/shared-with-portable-xnnpack.md` "NOT shared" table.
Related GPU-int64 surface: `vulkan-int64-truncation.md` (separate subagent).
