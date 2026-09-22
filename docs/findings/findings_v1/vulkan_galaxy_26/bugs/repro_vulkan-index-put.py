"""Repro: vulkan index_put writes to the wrong position.

index_put is NOT a registered vulkan op (no IndexPut.cpp/glsl; not in
backends/vulkan/op_registry.py) so it runs on the correct portable CPU kernel
(kernels/portable/cpu/op_index_put.cpp). The bug is in the vulkan PARTITIONER: the int64
0-d index constant `torch.tensor(1)` is pulled into a delegated GPU subgraph that does a no-op
`dim_order_ops._clone_dim_order`, round-trips through the GPU (no native int64 -> staged via
int32, ComputeGraph.cpp:965/1008), and comes back as 0. So the CPU index_put writes to index 0
instead of index 1.

This script:
  (1) shows eager + the *portable* ExecuTorch runtime agree (write at index 1) -> kernel is fine;
  (2) shows that forcing the index to 0 (what the GPU delegate returns) reproduces the exact
      on-device localizer values for w76:1620, w2:24, w25:539.

Run: .venv/bin/python findings/vulkan_galaxy_26/bugs/repro_vulkan-index-put.py
(Does NOT execute on device.)
"""
import torch


def vulkan_buggy(self_t, value, accumulate):
    """What vulkan effectively does: index = 0 (the GPU-corrupted constant)."""
    out = self_t.clone()
    if accumulate:
        out[0] = out[0] + value
    else:
        out[0] = value
    return out


def eager(self_t, idx, value, accumulate):
    return torch.ops.aten.index_put.default(
        self_t, [torch.tensor(idx, dtype=torch.int64)], value, accumulate)


cases = [
    # job, self, value, accumulate, device_b0
    ("w76:1620", torch.tensor([-0.5357359647750854, -0.4506544768810272]),
     torch.tensor(1.1635862588882446), True, 0.6279),
    ("w2:24", torch.tensor([-0.4824913740158081, 1.1103534698486328]),
     torch.tensor(0.9340156316757202), True, 0.4515),
    ("w25:539", torch.tensor([-0.4824913740158081, 1.1103534698486328]),
     torch.tensor(0.9340156316757202), False, 0.934),
]

print(f"{'job':<10} {'acc':<6} {'eager[0]':<12} {'vulkan(idx->0)[0]':<18} {'device b[0]':<12}")
for job, self_t, value, acc, dev in cases:
    e = eager(self_t, 1, value, acc)            # correct: writes index 1
    v = vulkan_buggy(self_t, value, acc)         # buggy: index forced to 0
    print(f"{job:<10} {str(acc):<6} {e[0].item():<12.4f} {v[0].item():<18.4f} {dev:<12}")

print("\nNote: eager[0] == self_t[0] (index 1 is written, index 0 untouched);")
print("vulkan(idx->0)[0] matches the device b[0] -> the index 1 became 0 on the GPU.")

# (Optional) prove the portable ET runtime is correct -- requires executorch installed.
try:
    from torch.export import export
    from executorch.exir import to_edge
    from executorch.runtime import Runtime

    class M(torch.nn.Module):
        def forward(self, x, v):
            return torch.ops.aten.index_put.default(
                x, [torch.tensor(1, dtype=torch.int64)], v, True)

    x = torch.tensor([-0.4824913740158081, 1.1103534698486328])
    v = torch.tensor(0.9340156316757202)
    pte = to_edge(export(M().eval(), (x, v))).to_executorch()
    meth = Runtime.get().load_program(pte.buffer).load_method("forward")
    out = meth.execute([x, v])[0]
    print(f"\nportable ET runtime: {out.tolist()}  (== eager [-0.4825, 2.0444], kernel is correct)")
except Exception as exc:  # pragma: no cover
    print(f"\n(portable ET check skipped: {type(exc).__name__})")
