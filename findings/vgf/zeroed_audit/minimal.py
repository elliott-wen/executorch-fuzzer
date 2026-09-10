import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
from executorch.exir._serialize._program import deserialize_pte_binary
SZ={0:1,1:1,2:2,3:4,4:8,5:2,6:4,7:8,11:1}

def mk(body, ret, leafdecl="L0 = torch.randn((4,), dtype=torch.float32)", leaves="L0"):
    return ("import torch\ntorch.manual_seed(7)\n"+leafdecl+"\n"
            + f"def g({leaves}):\n" + "".join(f"    {b}\n" for b in body)
            + f"    return ({ret},)\n" + f"LEAVES = [{leaves}]\n")

def plan(pte):
    prog=deserialize_pte_binary(pte); prog=getattr(prog,"program",prog); p=prog.execution_plan[0]
    def rr(i):
        v=p.values[i].val
        if type(v).__name__!="Tensor": return None
        n=1
        for d in v.sizes: n*=d
        ai=v.allocation_info
        return (i,(ai.memory_id if ai else None),(ai.memory_offset if ai else None),n*SZ.get(v.scalar_type,4))
    instr=[]
    for ins in (p.chains[0].instructions or []):
        it=ins.instr_args
        instr.append((type(it).__name__[:4],(p.operators[it.op_index].name.split("::")[-1] if type(it).__name__=="KernelCall" else ""),list(getattr(it,"args",[]) or [])))
    return [rr(i) for i in p.inputs],[rr(i) for i in p.outputs],instr

def probe(name, src, nrep=3):
    try: j=build_job(src,"vgf",False)
    except Exception as e:
        print(f"{name:40s} BUILD_EXC {type(e).__name__}: {str(e)[:80]}"); return None
    if j.status!="READY":
        print(f"{name:40s} BUILD:{j.status} {str(j.detail)[:70]}"); return None
    pin,pout,instr=plan(j.pte)
    verds=[]
    for _ in range(nrep):
        bufs,err=run_pte(j,"min")
        if bufs is None: verds.append("RUN:"+err); continue
        ts,fb,si=to_tensors(bufs,j.eager,j.user_pos)
        row=[]
        for k,t in enumerate(ts):
            e=j.eager[k]
            if t.numel()!=e.numel(): row.append("NUMEL"); continue
            d=t.reshape(e.shape); st,_=cmp._cmp(e,d,k)
            if st=="OK": row.append("ok")
            elif d.float().abs().max()<1e-6 and e.float().abs().max()>1e-3: row.append("ZERO")
            else: row.append("bad")
        verds.append(",".join(row))
    print(f"{name:40s} deleg={j.delegated_ops:2d} up={j.user_pos} {'  '.join(str(v) for v in verds)}")
    print(f"{'':40s} in={pin} out={pout} instr={instr}")
    return verds

A="torch.ops.aten"
F16="L0 = torch.randn((2,4), dtype=torch.float16)"
F32="L0 = torch.randn((2,4), dtype=torch.float32)"
CASES=[]
for tag,decl in (("f32",F32),("f16",F16)):
  CASES += [
   (f"[{tag}] 1out relu",            mk([f"n0 = {A}.relu.default(L0)"],"n0",decl)),
   (f"[{tag}] 2out relu,abs",        mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.abs.default(L0)"],"n0, n1",decl)),
   (f"[{tag}] 2out abs,relu",        mk([f"n0 = {A}.abs.default(L0)",f"n1 = {A}.relu.default(L0)"],"n0, n1",decl)),
   (f"[{tag}] 2out relu,relu (dup)", mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.relu.default(L0)"],"n0, n1",decl)),
   (f"[{tag}] 2out same node twice", mk([f"n0 = {A}.relu.default(L0)"],"n0, n0",decl)),
   (f"[{tag}] 3out relu,abs,neg",    mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.abs.default(L0)",f"n2 = {A}.neg.default(L0)"],"n0, n1, n2",decl)),
   (f"[{tag}] 4out",                 mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.abs.default(L0)",f"n2 = {A}.neg.default(L0)",f"n3 = {A}.sigmoid.default(L0)"],"n0, n1, n2, n3",decl)),
   (f"[{tag}] 2out chain relu,abs(n0)", mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.abs.default(n0)"],"n0, n1",decl)),
   (f"[{tag}] 2out relu + clone(L0)",mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.clone.default(L0)"],"n0, n1",decl)),
   (f"[{tag}] 2out relu + view(L0)", mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.view_copy.default(L0, [4,2])"],"n0, n1",decl)),
  ]
for n,s in CASES: probe(n,s)
