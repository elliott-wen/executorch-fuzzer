# broadcast_to / expand_copy with all-zero (empty/scalar) target — shape divergence

**Signature:** root op `broadcast_to` — `shape`
**Cluster size:** ~14 born-here (localizer), all flagged via a downstream reduce/index op that
takes the broadcast result as its `out=` buffer. Flagged-op carriers: `min` (w0:1007, w21:273,
w26:427, w26:272, w3:25, w34:1443, w28:1478, w31:1503, w65:987, w44:1154, w53:923),
`topk` (w0:1408, w15:772, w26:1220, w23:623, w17:1665, w28:1417), `max` (w31:1503),
`log10` (w38:985), `logical_and` (w3:1547), `cos` (w44:1154).
**Classification:** `generator-artifact` (degenerate empty/scalar broadcast + localizer
out-buffer aliasing). Vulkan's broadcast shape is actually *correct*; the recorded "eager" shape
is a mutated value, not the broadcast's true output.

## What happens
The corpus emits `broadcast_to(x, [0,0,0,0,0])` (or `[0]`) where the target shape is literally
all zeros — a degenerate, 0-numel target. The result is then handed to a reduction op as its
preallocated `out=` buffer, e.g. `min.dim_min(L2, 0, False, min=n15, min_indices=...)`.

example job `w0:1007` (root n15):
- eager (localizer): `()`        — n15 *after* the downstream `min` resized it in place
- vulkan (device):  `(0,0,0,0,0)` — n15 as the GPU expand kernel actually produced it

example `w26:427` (root n16, with an empty upstream): eager `(1,)` vs vulkan `(0,)`.
Here n5 = `slice_copy(n3,0,0,0,1)` is a length-0 slice → empty `(0,)`; `broadcast_to(n15,[0])`
gives `(0,)`, then `min` resizes its `out=` to `(1,)` in eager only.

The broadcast itself is NOT computed wrong by Vulkan. Run in isolation, eager agrees with the
device exactly:
```
torch.ops.aten.broadcast_to(torch.zeros(1,dtype=torch.int64), [0,0,0,0,0]).shape == (0,0,0,0,0)
torch.ops.aten.broadcast_to(torch.zeros(1,dtype=torch.int64), [0]).shape         == (0,)
```
So vulkan `(0,0,0,0,0)` / `(0,)` is the *correct* logical torch shape for the broadcast node.

## Root cause
Two independent things combine; neither is a wrong-value bug in the broadcast kernel:

1. **The recorded "eager" shape is an aliased, post-mutation value.** The all-node localizer
   (`gen/divergence.py:146 _load_allnode_module`) rewrites `g` to `return (n0, n1, ... nK)` and
   captures every node *after the whole graph has run*. PyTorch's out-variant
   `aten.min.dim_min(..., min=n15, ...)` resizes its `out=` tensor in place to the true reduction
   output shape and returns the *same object* (`out[0] is n15`). So once `min` runs, the Python
   name `n15` no longer holds the `(0,0,0,0,0)` broadcast result — it holds the resized `()`
   reduction output. The localizer then compares this mutated `()` against the device's
   un-mutated broadcast tensor `(0,0,0,0,0)` and flags a "shape" divergence at the broadcast.
   (Demonstrated: `n15` is `(0,0,0,0,0)` before `min`, `()` after — see repro.)

2. **Vulkan does not perform the same in-place out-resize.** `broadcast_to` is decomposed to
   `aten.expand_copy.default` (no `broadcast_to` op exists in the partitioner / runtime). The
   expand kernel computes its output sizes straight from the (all-zero) target list:
   `pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/impl/Expand.cpp:38-48`
   ```cpp
   std::vector<int64_t> out_sizes(target_sizes.size());
   for (size_t i = 0; i < target_sizes.size(); i++) {
     ...
     else { out_sizes[i] = target_sizes[i]; }   // 0 stays 0
   }
   graph->virtual_resize(out, out_sizes);        // -> (0,0,0,0,0)
   ```
   and `op_registry.py:1291` marks `expand_copy` `supports_resize=False`. The reduce op's `out=`
   tensor on Vulkan keeps the broadcast's `(0,0,0,0,0)` metadata; there is no in-place resize of
   the supplied `min=` buffer to the reduction's logical shape the way eager's out-variant does.
   So the device faithfully reports `(0,0,0,0,0)` / `(0,)` via
   `ComputeGraph::sizes_of` (`runtime/graph/ComputeGraph.cpp:317-320`,
   `out_sizes` set in `Expand.cpp:48`).

In other words: the broadcast node's value is identical on both sides; the apparent divergence
is the localizer comparing eager's *downstream-mutated* alias against the device's *raw* broadcast
output. The underlying graph is degenerate (a 0-numel / scalar broadcast whose result is only
ever consumed as a throwaway `out=` buffer), so the comparison is ill-posed.

## Minimal repro
`repro_vulkan-broadcast-shape.py` — shows (a) eager `broadcast_to(.,[0,0,0,0,0])` == device
`(0,0,0,0,0)` (broadcast is correct), and (b) the downstream `min(out=...)` mutates the same
tensor object to `()` in eager, which is what the localizer records — proving the "shape"
divergence is an out-buffer aliasing artifact, not a wrong vulkan value.

## Notes
Vulkan-specific in surface (only vulkan's expand path + the localizer's eager-side mutation
produce the mismatch) but the underlying cause is a generator artifact: degenerate all-zero
broadcast targets, often fed by upstream empty tensors (`slice_copy(...,0,0,...)` length-0,
`narrow_copy(...,0,0,0)`), used purely as reduction `out=` buffers. Not a numerical bug. If the
generator stopped emitting all-zero broadcast targets / 0-numel reduction out-buffers, or if the
localizer captured node values immediately (not after the full graph mutates shared out-tensors),
the signature disappears. No correct vulkan kernel would change the answer.
