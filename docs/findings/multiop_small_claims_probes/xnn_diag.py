import os,sys,warnings,logging,json
os.environ.setdefault("CUDA_VISIBLE_DEVICES","");os.environ["MOBILE_BACKENDS"]="xnnpack"
sys.path.insert(0,"/data/jwen929"); warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.gen.export import build_job
from mobile.executor.et_runner import run_pte
from mobile.executor import compare as cmp
from mobile.gen.diff.cone_predicate import output_set_localizer
CO="/data/jwen929/mobile/corpus_v1/xnnpack"
def show(src,label,reps=5):
    j=build_job(src,"xnnpack",False)
    if j.status!="READY": print(f"  {label}: {j.status} {j.detail[:80]}"); return
    print(f"  {label}: deleg={j.delegated_ops} n_eager={len(j.eager)}")
    for r in range(reps):
        out=run_pte(j.pte,j.inputs); et=list(out)
        if len(et)>len(j.eager): et=et[len(et)-len(j.eager):]
        res=[]
        for i,(e,d) in enumerate(zip(j.eager,et)):
            s,det=cmp._cmp(e,d,i); res.append(f"o{i}:{s}"+(f"({det.split('] ',1)[-1][:40]})" if s!="OK" else ""))
        print(f"    rep{r}: "+"  ".join(res))
    e=j.eager; out=run_pte(j.pte,j.inputs); et=list(out)
    if len(et)>len(e): et=et[len(et)-len(e):]
    for i,(a,b) in enumerate(zip(e,et)):
        print(f"    out{i} eager {tuple(a.shape)} {a.dtype} {[round(float(x),5) for x in a.flatten()[:5].tolist()]}")
        print(f"         et    {tuple(b.shape)} {b.dtype} {[round(float(x),5) for x in b.flatten()[:5].tolist()]}")
for jobfile,oi in [("w31/w31_508.py",7),("w9/w9_638.py",5)]:
    p=f"{CO}/{jobfile}"
    print("="*70); print(jobfile, "out", oi)
    src=open(p).read()
    print([l.strip() for l in src.splitlines() if l.strip().startswith("n") or l.strip().startswith("return")])
    others,build_out,target,ret=output_set_localizer(p,oi)
    print(f"  target={target} ret={ret}")
    show(build_out([target]),"A[target alone]")
    show(build_out(ret),"B[full return set]")
