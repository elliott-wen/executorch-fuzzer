#!/usr/bin/env python3
"""Is the real host-visible mechanism 'duplicate/aliased delegate outputs'?
Dump (a) the delegate-internal graph output list, (b) the serialized VkGraph
output value refs, (c) the OUTER .pte plan allocation for the delegate's
output slots."""
import os, sys, warnings, logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
import executorch.backends.vulkan.vulkan_preprocess as VP
from executorch.backends.vulkan.serialization.vulkan_graph_serialize import serialize_vulkan_graph

CAP = []
_orig_ser = VP.serialize_vulkan_graph
def traced_ser(vk_graph, *a, **kw):
    CAP.append(dict(inputs=list(vk_graph.input_ids), outputs=list(vk_graph.output_ids),
                    nvalues=len(vk_graph.values),
                    chains=[(c.node_id if hasattr(c,'node_id') else '?') for c in vk_graph.chain][:0]))
    return _orig_ser(vk_graph, *a, **kw)
VP.serialize_vulkan_graph = traced_ser

class Bug(torch.nn.Module):
    def forward(self, x):
        n0 = torch.ops.aten.sigmoid.default(x)
        n1 = torch.ops.aten.clone.default(n0)
        out0 = torch.ops.aten.prod.default(torch.ops.aten.acos.default(n0).to(torch.float16))
        out1 = torch.ops.aten.acosh.default(n1).to(torch.float16)
        return out0, out1
class Control(torch.nn.Module):
    def forward(self, x):
        n0 = torch.ops.aten.sigmoid.default(x)
        n1 = torch.ops.aten.add.Tensor(n0, 0.0)
        out0 = torch.ops.aten.prod.default(torch.ops.aten.acos.default(n0).to(torch.float16))
        out1 = torch.ops.aten.acosh.default(n1).to(torch.float16)
        return out0, out1
class DirectAlias(torch.nn.Module):
    """clone of a shared tensor co-returned as its own output (all-vulkan)."""
    def forward(self, x):
        n0 = torch.ops.aten.sigmoid.default(x)
        n1 = torch.ops.aten.clone.default(n0)
        n2 = torch.ops.aten.mul.Tensor(n0, 2.0)
        return n1, n2, n0

for name, M, x in [("Bug(clone)", Bug, torch.tensor([-0.48249137, 1.11035347])),
                   ("Control(add0)", Control, torch.tensor([-0.48249137, 1.11035347])),
                   ("DirectAlias", DirectAlias, torch.randn(4))]:
    CAP.clear()
    print(f"\n=== {name}")
    ep = torch.export.export(M(), (x,))
    exe = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False)).to_executorch()
    for i, c in enumerate(CAP):
        dup = len(c['outputs']) != len(set(c['outputs']))
        print(f"  VkGraph#{i}: inputs={c['inputs']} outputs={c['outputs']} nvalues={c['nvalues']}"
              f" {'*** DUPLICATE OUTPUT VALUE REF ***' if dup else ''}")
    # outer program
    from executorch.exir._serialize._program import deserialize_pte_binary
    prog = deserialize_pte_binary(bytes(exe.buffer))
    prog = getattr(prog, "program", prog)
    plan = prog.execution_plan[0]
    print(f"  OUTER plan: outputs={list(plan.outputs)}")
    # walk instructions: find delegate calls and their arg value ids
    for instr in plan.chains[0].instructions:
        k = type(instr.instr_args).__name__
        if k == "DelegateCall":
            print(f"    DelegateCall delegate_index={instr.instr_args.delegate_index} args={list(instr.instr_args.args)}")
        elif k == "KernelCall":
            op = prog.execution_plan[0].operators[instr.instr_args.op_index]
            print(f"    KernelCall {op.name}.{op.overload} args={list(instr.instr_args.args)}")
    def ai(i):
        v = plan.values[i].val
        if type(v).__name__ != "Tensor": return f"v{i}:{type(v).__name__}"
        a = v.allocation_info
        return (f"v{i}: sizes={list(v.sizes)} dtype={v.scalar_type} "
                f"alloc={'None(const/ext)' if a is None else f'mem{a.memory_id}@{a.memory_offset}'} "
                f"data_buffer_idx={v.data_buffer_idx}")
    seen = set()
    for instr in plan.chains[0].instructions:
        for arg in getattr(instr.instr_args, "args", []):
            if isinstance(arg, int) and arg not in seen:
                seen.add(arg); print("      " + ai(arg))
