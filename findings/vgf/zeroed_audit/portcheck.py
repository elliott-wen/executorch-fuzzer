import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
from mobile.net import et_runner
BR={}
with open("/data/jwen929/mobile/findings/vgf/bisect_results.tsv") as f:
    hdr=f.readline().rstrip("\n").split("\t"); ix={c:i for i,c in enumerate(hdr)}
    for line in f:
        r=line.rstrip("\n").split("\t")
        if len(r)<=ix["deleg_full"]: continue
        BR[(r[0],r[1])]=(r[ix["verdict"]],r[ix["needed_siblings"]],r[ix["needed_live"]])
for spec in ["w49:644#0","w74:450#3","w93:501#0","w60:389#1"]:
    job,oi=spec.split("#"); v,ns,nl=BR[(job,oi)]
    needed=[x for x in nl.split(",") if x]
    anc,bs,target=cone_localizer(C.job_file(CO,job,"py"),int(oi))
    src=bs(needed)
    out=[]
    for bk in ("vgf","portable","xnnpack"):
        try:
            j=build_job(src,bk,False)
        except Exception as e: out.append(f"{bk}=BUILD_EXC"); continue
        if j.status!="READY": out.append(f"{bk}=BUILD:{j.status}"); continue
        if bk=="vgf":
            bufs,err=run_pte(j,"pc")
            if bufs is None: out.append(f"vgf=RUN:{err}"); continue
            ts,fb,si=to_tensors(bufs,j.eager,j.user_pos)
            e=j.eager[-1]; d=ts[-1].reshape(e.shape); st,_=cmp._cmp(e,d,0)
            out.append(f"vgf(deleg={j.delegated_ops})={'OK' if st=='OK' else ('ZERO' if d.float().abs().max()<1e-6 else 'BAD')}")
        else:
            try:
                et=et_runner.run_pte(j.pte,j.inputs)
            except Exception as ex: out.append(f"{bk}=EXC:{type(ex).__name__}"); continue
            et=cmp.select(et,j.user_pos)
            if len(et)>len(j.eager): et=et[len(et)-len(j.eager):]
            e=j.eager[-1]; d=et[-1]; st,_=cmp._cmp(e,d,0)
            out.append(f"{bk}(deleg={j.delegated_ops})={'OK' if st=='OK' else ('ZERO' if d.float().abs().max()<1e-6 else 'BAD')}")
    print(job,oi,"target",target,"|","  ".join(out),flush=True)
