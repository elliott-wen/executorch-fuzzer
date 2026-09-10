import os,sys,warnings,logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0,"/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner

def dump(label, mod, args):
    print("\n"+"="*70); print(label)
    ep=torch.export.export(mod,args)
    ee=to_edge_transform_and_lower(ep, partitioner=[CoreMLPartitioner()])
    gm=ee.exported_program().graph_module
    for n in gm.graph.nodes:
        tgt=n.target if not callable(n.target) else getattr(n.target,'_name',getattr(n.target,'__name__',str(n.target)))
        print(f"  {n.op:16s} {str(n.name):24s} {str(tgt)[:60]:60s} args={[str(a) for a in n.args][:4]}")
    print("  output node args:", gm.graph.output_node().args)
    # graph signature output specs
    sig=ee.exported_program().graph_signature
    for s in sig.output_specs: print("   out_spec:",s.kind, s.arg, "target=",s.target)
    ep2=ee.to_executorch()
    p=ep2.exported_program()
    print("  --- after to_executorch ---")
    for n in p.graph_module.graph.nodes:
        if n.op=="output": print("   output:",n.args)
    return ee

class TwoAlias(torch.nn.Module):
    def forward(self,x):
        n2=torch.ops.aten.mul.Scalar(x,2.0)
        a=torch.ops.aten.lift_fresh_copy.default(n2)
        b=torch.ops.aten.alias_copy.default(n2)
        return (a,b)
class S4(torch.nn.Module):
    def forward(self,x):
        n2=torch.ops.aten.div.Scalar_mode(x,-4,rounding_mode=None)
        m=torch.ops.aten.max.default(n2)
        a=torch.ops.aten.lift_fresh_copy.default(n2)
        return (m,a)
x=torch.randn(4)
dump("s2_two_alias", TwoAlias(), (x,))
dump("s4_unsupported_src", S4(), (x,))
