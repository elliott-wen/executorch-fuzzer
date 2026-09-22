import os,sys,warnings,logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0,"/data/jwen929"); warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch, coremltools as ct
class V(torch.nn.Module):
    def __init__(s,c): super().__init__(); s.c=c
    def forward(s,x): return torch.ops.aten.var.correction(x,[0],correction=s.c,keepdim=False)
for c in (None,0,1):
    x=torch.randn(4)
    ep=torch.export.export(V(c).eval(),(x,)).run_decompositions({})
    print("\n===== correction=",c)
    print("  decomposed graph:", [str(n.target).split('.aten.')[-1] for n in ep.graph.nodes if n.op=="call_function"])
    try:
        m=ct.convert(ep,minimum_deployment_target=ct.target.iOS17,convert_to="mlprogram",
                     compute_precision=ct.precision.FLOAT32)
        for op in m._mil_program.functions["main"].operations:
            vals={}
            for k,v in op.inputs.items():
                try: vals[k]=v.val if getattr(v,'val',None) is not None else v.name
                except Exception: vals[k]=str(v)
            print("   MIL",op.op_type,vals)
    except Exception as e: print("   convert failed:",type(e).__name__,str(e)[:160])
