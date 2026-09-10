#!/usr/bin/env python3
"""Causal test: is the delegate output-arity shortfall caused by
RemoveRedundantOpsTransform deleting the copy? Neutralise the pass and re-check."""
import os,sys,warnings,logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES","");os.environ.setdefault("MOBILE_BACKENDS","vulkan")
sys.path.insert(0,"/data/jwen929"); warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
import executorch.backends.vulkan.vulkan_preprocess as VP
from executorch.exir._serialize._program import deserialize_pte_binary
REC=[]; _o=VP.serialize_vulkan_graph
def tr(g,*a,**k):
    REC.append((len(g.input_ids),len(g.output_ids))); return _o(g,*a,**k)
VP.serialize_vulkan_graph=tr
class Bug(torch.nn.Module):
    def forward(self,x):
        n0=torch.ops.aten.sigmoid.default(x); n1=torch.ops.aten.clone.default(n0)
        return (torch.ops.aten.prod.default(torch.ops.aten.acos.default(n0).to(torch.float16)),
                torch.ops.aten.acosh.default(n1).to(torch.float16))
def go(tag):
    REC.clear()
    exe=to_edge_transform_and_lower(torch.export.export(Bug(),(torch.tensor([-0.48,1.11]),)),
        partitioner=[VulkanPartitioner()],compile_config=EdgeCompileConfig(_check_ir_validity=False)).to_executorch()
    prog=deserialize_pte_binary(bytes(exe.buffer)); prog=getattr(prog,"program",prog)
    calls=[i.instr_args for i in prog.execution_plan[0].chains[0].instructions
           if type(i.instr_args).__name__=="DelegateCall"]
    print(f"  {tag}:")
    for (nin,nout),c in zip(REC,calls):
        slots=len(c.args)-nin
        print(f"    blob: declared_inputs={nin} declared_outputs={nout} callsite_args={len(c.args)} "
              f"-> output_slots={slots} {'*** SHORTFALL '+str(slots-nout)+' UNWRITTEN ***' if slots!=nout else 'OK'}")
print("=== as shipped")
go("RemoveRedundantOpsTransform ACTIVE")
# neutralise the pass
VP.RemoveRedundantOpsTransform.redundant_ops = set()   # delete nothing
print("=== pass neutralised")
go("RemoveRedundantOpsTransform DISABLED")
