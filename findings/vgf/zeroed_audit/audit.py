import os, sys, json, argparse, traceback
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *

def bisect_rows():
    out={}
    with open("/data/jwen929/mobile/findings/vgf/bisect_results.tsv") as f:
        hdr=f.readline().rstrip("\n").split("\t"); ix={c:i for i,c in enumerate(hdr)}
        for line in f:
            r=line.rstrip("\n").split("\t")
            if len(r)<=ix["deleg_full"]: continue
            out[(r[0],r[1])]=(r[ix["verdict"]], r[ix["needed_siblings"]], r[ix["needed_live"]],
                              r[ix["deleg_full"]], r[ix["target"]], r[ix["n_out"]])
    return out

BR=bisect_rows()
NREP=int(os.environ.get("NREP","3"))

def do_sibling(job,out_idx,needed):
    others,bos,target,ret=output_set_localizer(C.job_file(CO,job,"py"),out_idx)
    srcA=bos([target]); sel=[o for o in ret if o in set([target]+needed)]; posB=sel.index(target)
    srcB=bos([target]+needed)
    return srcA,0,srcB,posB,target,sel
def do_comp(job,out_idx,needed):
    anc,build_src,target=cone_localizer(C.job_file(CO,job,"py"),out_idx)
    srcA=build_src([]); srcB=build_src(needed)
    return srcA,None,srcB,None,target,None   # pos = last output

def side(src,tag,pos,nrep):
    """returns dict"""
    res={"verdicts":[],"deleg":None,"user_pos":None,"n_out_files":None,"fellback":None,
         "eager_absmax":None,"dev_absmax":[],"n_eager":None,"build":None}
    for rep in range(nrep):
        try:
            j=build_job(src,"vgf",False)
        except Exception as e:
            res["verdicts"].append("BUILD_EXC:"+type(e).__name__); continue
        if j.status!="READY":
            res["build"]=f"{j.status}:{str(j.detail)[:40]}"; res["verdicts"].append("BUILD:"+j.status); continue
        res["deleg"]=j.delegated_ops; res["user_pos"]=list(j.user_pos) if j.user_pos else None
        res["n_eager"]=len(j.eager)
        p = pos if pos is not None else len(j.eager)-1
        res["eager_absmax"]=float(j.eager[p].float().abs().max()) if j.eager[p].numel() else 0.0
        bufs,err=run_pte(j,tag)
        if bufs is None: res["verdicts"].append("RUN:"+err); continue
        res["n_out_files"]=len(bufs)
        ts,fb,selidx=to_tensors(bufs,j.eager,j.user_pos); res["fellback"]=fb
        try:
            v,det,d=verdict_at(j.eager,ts,p)
        except Exception as e:
            res["verdicts"].append("CMP_EXC:"+type(e).__name__); continue
        res["verdicts"].append(v)
        if d is not None: res["dev_absmax"].append(float(d.float().abs().max()))
    return res

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cases",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--shard",type=int,default=0); ap.add_argument("--nshards",type=int,default=1)
    a=ap.parse_args()
    cases=[l.split() for l in open(a.cases) if l.strip()]
    mine=[c for i,c in enumerate(cases) if i%a.nshards==a.shard]
    done=set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add((json.loads(l)["job"],json.loads(l)["out"]))
            except: pass
    fh=open(a.out,"a",buffering=1)
    for job,oi in mine:
        if (job,oi) in done: continue
        rec={"job":job,"out":oi}
        try:
            v,ns,nl,df,tgt,nout=BR[(job,oi)]
            rec["verdict"]=v; rec["deleg_full"]=df; rec["n_out_orig"]=nout
            needed=[x for x in (ns if v=="SIBLING_DEPENDENT" else nl).split(",") if x]
            rec["needed"]=needed
            if v=="SIBLING_DEPENDENT":
                srcA,pA,srcB,pB,target,sel=do_sibling(job,int(oi),needed)
            else:
                srcA,pA,srcB,pB,target,sel=do_comp(job,int(oi),needed)
            rec["target"]=target; rec["tgt_op"]=node_ops(job).get(target,"?")
            rec["trigger_ops"]=[node_ops(job).get(s,"?") for s in needed]
            rec["A"]=side(srcA,"A",pA,NREP)
            rec["B"]=side(srcB,"B",pB,NREP)
            rec["posB"]=pB
        except Exception as e:
            rec["error"]=f"{type(e).__name__}:{str(e)[:120]}"
        fh.write(json.dumps(rec)+"\n")
        print(job,oi,rec.get("tgt_op"),"A=",rec.get("A",{}).get("verdicts"),"B=",rec.get("B",{}).get("verdicts"),rec.get("error",""),flush=True)

if __name__=="__main__": main()
