"""For each COMPOSITIONAL ZEROED case: the B graph = target + live ancestors, single output.
Test whether each individual live ANCESTOR, returned ALONE, is already wrong on the device.
If yes, the case is an ancestor-operator bug, not a graph-optimization bug."""
import os,sys,ast,json,argparse
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *

def retvar(src, name):
    m=ast.parse(src)
    for st in m.body:
        if isinstance(st,ast.FunctionDef) and st.name=="g":
            for i,s in enumerate(st.body):
                if isinstance(s,ast.Return): st.body=st.body[:i]; break
            st.body.append(ast.parse(f"return ({name},)").body[0])
    return ast.unparse(m)

def runone(src,nrep=3):
    try: j=build_job(src,"vgf",False)
    except Exception as e: return "BUILD_EXC:"+type(e).__name__,None
    if j.status!="READY": return "BUILD:"+j.status,None
    vs=[]
    for _ in range(nrep):
        bufs,err=run_pte(j,"cp")
        if bufs is None: vs.append("RUN:"+err); continue
        ts,fb,si=to_tensors(bufs,j.eager,j.user_pos)
        p=len(j.eager)-1; e=j.eager[p]; t=ts[p]
        if t.numel()!=e.numel(): vs.append("NUMEL"); continue
        d=t.reshape(e.shape); st,_=cmp._cmp(e,d,p)
        if st=="OK": vs.append("ok")
        elif e.numel() and d.float().abs().max()<1e-6 and e.float().abs().max()>1e-3: vs.append("ZERO")
        else: vs.append("bad")
    return ",".join(vs), j.delegated_ops

BR={}
with open("/data/jwen929/mobile/findings/vgf/bisect_results.tsv") as f:
    hdr=f.readline().rstrip("\n").split("\t"); ix={c:i for i,c in enumerate(hdr)}
    for line in f:
        r=line.rstrip("\n").split("\t")
        if len(r)<=ix["deleg_full"]: continue
        BR[(r[0],r[1])]=(r[ix["verdict"]],r[ix["needed_siblings"]],r[ix["needed_live"]])
ap=argparse.ArgumentParser(); ap.add_argument("--cases"); ap.add_argument("--out"); ap.add_argument("--shard",type=int,default=0); ap.add_argument("--nshards",type=int,default=1)
a=ap.parse_args()
cases=[l.split() for l in open(a.cases) if l.strip()]
mine=[c for i,c in enumerate(cases) if i%a.nshards==a.shard]
fh=open(a.out,"a",buffering=1)
for job,oi in mine:
    v,ns,nl=BR[(job,oi)]
    if v!="COMPOSITIONAL": continue
    needed=[x for x in nl.split(",") if x]
    anc,build_src,target=cone_localizer(C.job_file(CO,job,"py"),int(oi))
    srcB=build_src(needed)
    rec={"job":job,"out":oi,"target":target,"needed":needed,"anc":{}}
    rec["B"],rec["Bdeleg"]=runone(srcB)
    for nm in needed:
        try: rec["anc"][nm]=runone(retvar(srcB,nm))
        except Exception as e: rec["anc"][nm]=("EXC:"+type(e).__name__,None)
    fh.write(json.dumps(rec)+"\n")
    print(job,oi,"B=",rec["B"],"anc=",{k:x[0] for k,x in rec["anc"].items()},flush=True)
