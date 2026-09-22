#!/usr/bin/env python3
"""Order-independent (sound) version of the arity-shortfall scan: compare, PER JOB,
sum(declared VkGraph outputs) against sum(call-site output slots). A deficit means at
least one delegate blob leaves an output buffer unwritten, regardless of blob ordering."""
import os, sys, warnings, logging, random, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); os.environ.setdefault("MOBILE_BACKENDS","vulkan")
sys.path.insert(0,"/data/jwen929"); warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from pathlib import Path
import executorch.backends.vulkan.vulkan_preprocess as VP
from mobile.gen.export import build_job
from executorch.exir._serialize._program import deserialize_pte_binary
REC=[]; _o=VP.serialize_vulkan_graph
def tr(g,*a,**k):
    REC.append((len(g.input_ids),len(g.output_ids))); return _o(g,*a,**k)
VP.serialize_vulkan_graph=tr
N=int(sys.argv[1]); CO=sys.argv[2]
files=sorted(Path(CO).glob("w*/*.py")); random.seed(4242); random.shuffle(files)
st=dict(tried=0,ready=0,notready=0,exc=0,jobs_with_deleg=0,deficit_jobs=0,deficit_slots=0,total_slots=0)
ex=[]
for f in files:
    if st["tried"]>=N: break
    st["tried"]+=1; REC.clear()
    try: j=build_job(f.read_text(),"vulkan",False)
    except Exception: st["exc"]+=1; continue
    if j.status!="READY": st["notready"]+=1; continue
    st["ready"]+=1
    try:
        prog=deserialize_pte_binary(j.pte); prog=getattr(prog,"program",prog)
        calls=[i.instr_args for i in prog.execution_plan[0].chains[0].instructions
               if type(i.instr_args).__name__=="DelegateCall"]
    except Exception: st["exc"]+=1; continue
    if not calls or len(calls)!=len(REC): continue
    st["jobs_with_deleg"]+=1
    decl=sum(o for _,o in REC)
    slots=sum(len(c.args) for c in calls)-sum(i for i,_ in REC)
    st["total_slots"]+=slots
    if slots>decl:
        st["deficit_jobs"]+=1; st["deficit_slots"]+=slots-decl
        if len(ex)<20: ex.append((f.name,decl,slots))
    if st["tried"]%25==0: print(json.dumps(st),flush=True)
print("FINAL",json.dumps(st))
for e in ex: print("EX",e)
