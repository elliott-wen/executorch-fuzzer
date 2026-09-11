import os,sys,argparse,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
from mobile.executor import et_runner
BR={}
with open("/data/jwen929/mobile/findings/vgf/bisect_results.tsv") as f:
    hdr=f.readline().rstrip("\n").split("\t"); ix={c:i for i,c in enumerate(hdr)}
    for line in f:
        r=line.rstrip("\n").split("\t")
        if len(r)<=ix["deleg_full"]: continue
        BR[(r[0],r[1])]=(r[ix["verdict"]],r[ix["needed_siblings"]],r[ix["needed_live"]])
ap=argparse.ArgumentParser(); ap.add_argument("--cases"); ap.add_argument("--shard",type=int,default=0); ap.add_argument("--nshards",type=int,default=1)
a=ap.parse_args()
cases=[l.split() for l in open(a.cases) if l.strip()]
for i,(job,oi) in enumerate(cases):
    if i%a.nshards!=a.shard: continue
    v,ns,nl=BR[(job,oi)]
    if v!="SIBLING_DEPENDENT": continue
    needed=[x for x in ns.split(",") if x]
    others,bos,target,ret=output_set_localizer(C.job_file(CO,job,"py"),int(oi))
    sel=[o for o in ret if o in set([target]+needed)]; p=sel.index(target)
    src=bos([target]+needed); out=[]
    for bk in ("portable","xnnpack"):
        try: j=build_job(src,bk,False)
        except Exception: out.append(f"{bk}=EXC"); continue
        if j.status!="READY": out.append(f"{bk}=BUILD:{j.status}"); continue
        try: et=et_runner.run_pte(j.pte,j.inputs)
        except Exception as ex: out.append(f"{bk}=RUNEXC"); continue
        et=cmp.select(et,j.user_pos)
        if len(et)>len(j.eager): et=et[len(et)-len(j.eager):]
        e=j.eager[p]; d=et[p]; st,_=cmp._cmp(e,d,p)
        z = (d.numel()>0 and d.float().abs().max()<1e-6)
        out.append(f"{bk}(d={j.delegated_ops})={'OK' if st=='OK' else ('ZERO' if z else 'BAD')}")
    print(job,oi,target,"|","  ".join(out),flush=True)
