import os,sys,warnings,logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0,"/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from mobile.gen.export import build_job
from mobile.gen.export.backends import get_backend
from executorch.exir._serialize._program import deserialize_pte_binary

print("coreml backend available on this host:", get_backend("coreml") is not None)

HEAD='''import torch
torch.manual_seed(0)
L0 = torch.randn(4)
def g(L0):
'''
PROBES={
 "s2_two_alias":  "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    a = torch.ops.aten.lift_fresh_copy.default(n2)\n    b = torch.ops.aten.alias_copy.default(n2)\n    return (a, b)\n",
 "s2_self_plus_alias": "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    b = torch.ops.aten.alias_copy.default(n2)\n    return (n2, b)\n",
 "s2_leaf_two_alias": "    a = torch.ops.aten.alias_copy.default(L0)\n    b = torch.ops.aten.alias_copy.default(L0)\n    return (a, b)\n",
 "s2_three_alias": "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    a = torch.ops.aten.lift_fresh_copy.default(n2)\n    b = torch.ops.aten.alias_copy.default(n2)\n    c = torch.ops.aten.view_copy.default(n2, [4])\n    return (a, b, c)\n",
 "s2_diff_tensors_CONTROL": "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    n3 = torch.ops.aten.add.Scalar(L0, 1.0)\n    a = torch.ops.aten.alias_copy.default(n2)\n    b = torch.ops.aten.alias_copy.default(n3)\n    return (a, b)\n",
 "s2_neutralized_plus0": "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    a = torch.ops.aten.lift_fresh_copy.default(n2)\n    b = torch.ops.aten.add.Scalar(torch.ops.aten.alias_copy.default(n2), 0.0)\n    return (a, b)\n",
 "s4_unsupported_src": "    n2 = torch.ops.aten.div.Scalar_mode(L0, -4, rounding_mode=None)\n    m = torch.ops.aten.max.default(n2)\n    a = torch.ops.aten.lift_fresh_copy.default(n2)\n    return (m, a)\n",
 "s4_supported_src_CONTROL": "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    m = torch.ops.aten.max.default(n2)\n    a = torch.ops.aten.lift_fresh_copy.default(n2)\n    return (m, a)\n",
}
BK=os.environ.get("BK","coreml")
for name,body in PROBES.items():
    src=HEAD+body+"LEAVES = [L0]\n"
    print("\n"+"="*72); print(name)
    try: j=build_job(src,BK)
    except Exception as e: print("  BUILD_EXC",type(e).__name__,str(e)[:200]); continue
    if j.status!="READY": print("  BUILD:",j.status,str(j.detail)[:200]); continue
    print(f"  delegated_ops={getattr(j,'delegated_ops',None)} user_pos={j.user_pos} n_eager={len(j.eager)}")
    prog=deserialize_pte_binary(j.pte); prog=getattr(prog,"program",prog)
    plan=prog.execution_plan[0]
    print(f"  plan.outputs (value idx) = {list(plan.outputs)}   plan.inputs={list(plan.inputs)}")
    seen={}
    for oi,vi in enumerate(plan.outputs):
        v=plan.values[vi].val
        tn=type(v).__name__
        if tn!="Tensor": print(f"   out[{oi}] valueidx={vi} {tn}"); continue
        ai=v.allocation_info
        key=(getattr(ai,'memory_id',None),getattr(ai,'memory_offset_low',getattr(ai,'memory_offset',None)))
        print(f"   out[{oi}] valueidx={vi} sizes={list(v.sizes)} dtype={v.scalar_type} "
              f"data_buffer_idx={v.data_buffer_idx} alloc={ai} key={key}")
        seen.setdefault(key,[]).append(oi)
    dupes={k:v for k,v in seen.items() if len(v)>1 and k[0] is not None}
    dup_vi = len(set(plan.outputs))!=len(plan.outputs)
    print(f"  >>> SAME value-index reused across outputs: {dup_vi}")
    print(f"  >>> DISTINCT outputs sharing (memory_id,offset): {dupes if dupes else 'NONE — plan gives every output its own buffer'}")
    # also dump the operator/delegate structure
    ops=[prog.operators[c.op_index].name if hasattr(c,'op_index') else '?' for chain in [plan.chains[0]] for c in [] ]
    try:
        names=[o.name for o in prog.operators]
        print(f"  operators in plan: {names}")
        print(f"  n_delegates: {len(plan.delegates)} -> {[d.id for d in plan.delegates]}")
    except Exception as e: print("  (opdump fail)",e)
