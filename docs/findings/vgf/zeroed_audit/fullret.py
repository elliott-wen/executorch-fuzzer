"""Re-run the report's plan_probe geometry (FULL-return graph, bos(ret)) but with a
liveness-aware overlap test. Prints the report's category and the corrected one."""
import os,sys,collections
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
from executorch.exir._serialize._program import deserialize_pte_binary
SZ={0:1,1:1,2:2,3:4,4:8,5:2,6:4,7:8,11:1}
REPORT_SZ={0:1,1:2,2:2,3:4,4:8,5:2,6:4,7:8,8:8,11:1}   # the buggy table in vgf_plan_probe.py

def load(job,oi):
    others,bos,target,ret=output_set_localizer(C.job_file(CO,job,"py"),int(oi))
    j=build_job(bos(ret),"vgf",False)
    if j.status!="READY": return None
    prog=deserialize_pte_binary(j.pte); prog=getattr(prog,"program",prog); p=prog.execution_plan[0]
    pos=ret.index(target)
    return p,pos,target

def cats(p,pos,sz):
    def rng(i):
        v=p.values[i].val
        if type(v).__name__!="Tensor": return None
        ai=v.allocation_info
        if ai is None: return None
        n=1
        for d in v.sizes: n*=d
        return (ai.memory_id,ai.memory_offset,ai.memory_offset+n*sz.get(v.scalar_type,4))
    outs=list(p.outputs); tv=outs[pos]; tr=rng(tv)
    if tr is None: return "NO_ALLOC","NO_ALLOC"
    instrs=[]
    for ins in (p.chains[0].instructions or []):
        it=ins.instr_args
        instrs.append((type(it).__name__,list(getattr(it,"args",[]) or [])))
    inset=set(p.inputs); outset=set(outs)
    first={};last={}
    for k,(kind,args) in enumerate(instrs):
        for a in args: first.setdefault(a,k); last[a]=k
    N=len(instrs)
    def live(v):
        f=-1 if v in inset else first.get(v,0)
        l=N if v in outset else last.get(v,f)
        return (f,l)
    tl=live(tv)
    prod="?"
    for kind,args in instrs:
        if tv in args: prod="delegate" if kind=="DelegateCall" else "kernel"; break
    rep=None; live_ov=[]; dead_ov=[]
    for i in range(len(p.values)):
        if i==tv: continue
        r=rng(i)
        if not r or r[0]!=tr[0] or not (r[1]<tr[2] and tr[1]<r[2]): continue
        kindtag="INPUT" if i in inset else ("OUTPUT" if i in outset else "INTERMEDIATE")
        if rep is None and i in inset: rep="OUT_ALIASES_INPUT"
        if rep is None and kindtag=="INTERMEDIATE": rep="OUT_ALIASES_INTERMEDIATE"
        wl=live(i)
        (live_ov if (wl[0]<=tl[1] and tl[0]<=wl[1]) else dead_ov).append((i,kindtag,wl,tl))
    reportcat=(rep or "OWN_BUFFER")+"/"+prod
    corrected=("LIVE_OVERLAP:"+str(live_ov[:2])) if live_ov else ("OWN_BUFFER(dead reuse)" if dead_ov else "OWN_BUFFER(no overlap)")
    return reportcat,corrected

jobs=[]
for l in open("/data/jwen929/mobile/findings/vgf/graphopt_all.tsv"):
    pp=l.rstrip("\n").split("\t")
    if len(pp)>=9 and pp[2]=="SIBLING_DEPENDENT" and pp[5]=="AB_OK" and pp[8]=="ZEROED": jobs.append((pp[0],pp[1]))
jobs=jobs[:60]
t=collections.Counter()
for job,oi in jobs:
    try:
        L=load(job,oi)
        if L is None: t["BUILD_FAIL"]+=1; continue
        p,pos,tgt=L
        a,b=cats(p,pos,REPORT_SZ); a2,b2=cats(p,pos,SZ)
        t[(a,b2.split(":")[0])]+=1
        print(f"{job:11s} out{oi}  report={a:35s} corrected={b2[:70]}",flush=True)
    except Exception as e:
        t["ERR"]+=1; print(job,oi,"ERR",str(e)[:70],flush=True)
print("\n=== report-category x corrected-category ===")
for k,v in t.most_common(): print(f"  {v:3d}  {k}")
