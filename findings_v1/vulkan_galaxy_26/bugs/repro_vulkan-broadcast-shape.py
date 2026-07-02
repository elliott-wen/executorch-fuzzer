#!/usr/bin/env python3
"""Repro: vulkan broadcast_to / expand_copy 'shape' divergence on degenerate
all-zero (empty/scalar) targets — example jobs w0:1007, w21:273, w26:427.

Classification: generator-artifact. Vulkan's broadcast shape is CORRECT (identical
to eager run in isolation). The recorded "shape" divergence is an out-buffer
aliasing artifact: a downstream reduce op (min.dim_min) takes the broadcast result
as its `out=` buffer and resizes that SAME tensor object in place in eager, while
the vulkan expand kernel (expand_copy, supports_resize=False) keeps the broadcast's
padded sizes. The localizer captures node values after the whole graph has run, so
it compares eager's mutated alias () against the device's raw broadcast (0,0,0,0,0).

Source refs:
  pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/impl/Expand.cpp:38-48
      (out_sizes copies the all-zero target verbatim -> (0,0,0,0,0))
  pytorch_ref/executorch/backends/vulkan/op_registry.py:1291
      (expand_copy supports_resize=False)
  pytorch_ref/executorch/backends/vulkan/runtime/graph/ComputeGraph.cpp:317-320
      (sizes_of faithfully returns those sizes to the host)
  gen/divergence.py:146 (_load_allnode_module returns every node AFTER g runs,
      so out= aliasing mutates the captured eager value)

Run: .venv/bin/python findings/vulkan_galaxy_26/bugs/repro_vulkan-broadcast-shape.py
"""
import torch

print("=== (a) broadcast_to in isolation: eager == device shape (vulkan is correct) ===")
for tgt in ([0, 0, 0, 0, 0], [0]):
    x = torch.zeros((1,), dtype=torch.int64)
    r = torch.ops.aten.broadcast_to.default(x, tgt)
    print(f"  broadcast_to((1,), {tgt}) -> eager {tuple(r.shape)}  "
          f"(device/vulkan reports the same; see Expand.cpp:48)")

print()
print("=== (b) the divergence is out-buffer aliasing, not a wrong value ===")
# w0:1007 shape: broadcast result n15 is used as the min= out buffer.
n15 = torch.ops.aten.broadcast_to.default(torch.zeros((1,), dtype=torch.int64),
                                          [0, 0, 0, 0, 0])
print("  n15 = broadcast_to(...)  before min :", tuple(n15.shape), "  id=", id(n15))
out = torch.ops.aten.min.dim_min(
    torch.zeros((1,), dtype=torch.int64), 0, False,
    min=n15, min_indices=torch.zeros((), dtype=torch.int64))
print("  n15  after min (resized in place)   :", tuple(n15.shape),
      "  min()[0] is n15 ->", out[0] is n15)
print("  => localizer records eager n15 = {}  vs  device n15 = (0, 0, 0, 0, 0)"
      .format(tuple(n15.shape)))
print("     i.e. 'eager() vs backend(0,0,0,0,0)' — vulkan kept the true broadcast shape;")
print("     eager's value was overwritten in place by the downstream min(out=...).")

print()
print("=== (c) w26:427 — same thing with an empty upstream tensor ===")
n5 = torch.zeros((0,), dtype=torch.int64)          # slice_copy(...,0,0,0,1) -> empty
n15b = torch.zeros((1,), dtype=torch.int64)        # to_copy(sqrt(...)) -> (1,)
n16 = torch.ops.aten.broadcast_to.default(n15b, [0])
print("  n16 = broadcast_to((1,), [0]) before min:", tuple(n16.shape))
n14 = torch.ops.aten.constant_pad_nd.default(
    torch.ops.aten.expand_copy.default(n5, [0], implicit=False), [0, 1], 0)
o = torch.ops.aten.min.dim_min(n14, 0, True, min=n16,
                               min_indices=torch.zeros((0,), dtype=torch.int64))
print("  n16 after min (resized in place)       :", tuple(n16.shape),
      " -> localizer: eager(1,) vs backend(0,)")
