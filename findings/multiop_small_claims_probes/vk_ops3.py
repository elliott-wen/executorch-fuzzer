#!/usr/bin/env python3
"""Which copy-ish ops, co-returned with a sibling, produce a Vulkan delegate
output-arity shortfall (an unwritten output slot)?"""
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
A=torch.ops.aten
OPS={
 "clone":                 lambda n0: A.clone.default(n0),
 "alias_copy":            lambda n0: A.alias_copy.default(n0),
 "expand_copy_same":      lambda n0: A.expand_copy.default(n0,[4,8]),
 "expand_copy_grow":      lambda n0: A.expand_copy.default(A.slice_copy.Tensor(n0,0,0,1,1),[4,8]),
 "view_copy":             lambda n0: A.view_copy.default(n0,[8,4]),
 "transpose_copy":        lambda n0: A.transpose_copy.int(n0,0,1),
 "permute_copy":          lambda n0: A.permute_copy.default(n0,[1,0]),
 "slice_copy_full":       lambda n0: A.slice_copy.Tensor(n0,0,0,4,1),
 "squeeze_copy_noop":     lambda n0: A.squeeze_copy.dim(n0,0),
 "to_same_dtype":         lambda n0: n0.to(torch.float32),
 "add0(control)":         lambda n0: A.add.Tensor(n0,0.0),
 "mul1(control)":         lambda n0: A.mul.Tensor(n0,1.0),
}
for name,f in OPS.items():
    class M(torch.nn.Module):
        def forward(self,x):
            n0=A.sigmoid.default(x); n1=f(n0)
            return A.prod.default(A.acos.default(n0)), A.acosh.default(n1)
    REC.clear()
    try:
        exe=to_edge_transform_and_lower(torch.export.export(M(),(torch.randn(4,8),)),
            partitioner=[VulkanPartitioner()],compile_config=EdgeCompileConfig(_check_ir_validity=False)).to_executorch()
    except Exception as e:
        print(f"{name:22s} LOWER_EXC {type(e).__name__}: {str(e)[:70]}"); continue
    prog=deserialize_pte_binary(bytes(exe.buffer)); prog=getattr(prog,"program",prog)
    calls=[i.instr_args for i in prog.execution_plan[0].chains[0].instructions
           if type(i.instr_args).__name__=="DelegateCall"]
    msgs=[]
    for (nin,nout),c in zip(REC,calls):
        slots=len(c.args)-nin
        msgs.append(f"decl_out={nout} slots={slots}"+(f" *** {slots-nout} UNWRITTEN ***" if slots!=nout else ""))
    print(f"{name:22s} " + " | ".join(msgs))
