import os, sys, json, argparse
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
from executorch.exir._serialize._program import deserialize_pte_binary

SZ={0:1,1:1,2:2,3:4,4:8,5:2,6:4,7:8,11:1,15:2,16:1}  # corrected element sizes

def plan_info(pte):
    prog=deserialize_pte_binary(pte); prog=getattr(prog,"program",prog)
    plan=prog.execution_plan[0]
    outs=[]
    for oi in plan.outputs:
        v=plan.values[oi].val
        if type(v).__name__!="Tensor": outs.append(None); continue
        n=1
        for d in v.sizes: n*=d
        es=SZ.get(v.scalar_type,4)
        ai=v.allocation_info
        outs.append({"val":oi,"sizes":list(v.sizes),"st":v.scalar_type,"numel":n,"bytes":n*es,
                     "mem_id":(ai.memory_id if ai else None),"off":(ai.memory_offset if ai else None),
                     "dbi":getattr(v,"data_buffer_idx",None)})
    ins=[]
    for ii in plan.inputs:
        v=plan.values[ii].val
        if type(v).__name__!="Tensor": ins.append(None); continue
        n=1
        for d in v.sizes: n*=d
        ai=v.allocation_info
        ins.append({"val":ii,"sizes":list(v.sizes),"bytes":n*SZ.get(v.scalar_type,4),
                    "mem_id":(ai.memory_id if ai else None),"off":(ai.memory_offset if ai else None)})
    instrs=[]
    for ins_ in (plan.chains[0].instructions or []):
        it=ins_.instr_args
        instrs.append({"k":type(it).__name__,"args":list(getattr(it,"args",[]) or []),
                       "op":(plan.operators[it.op_index].name if type(it).__name__=="KernelCall" else None)})
    # all planned tensor values
    allv=[]
    for i,vv in enumerate(plan.values):
        v=vv.val
        if type(v).__name__!="Tensor": continue
        ai=v.allocation_info
        if ai is None: continue
        n=1
        for d in v.sizes: n*=d
        allv.append({"val":i,"mem_id":ai.memory_id,"off":ai.memory_offset,
                     "bytes":n*SZ.get(v.scalar_type,4),"sizes":list(v.sizes)})
    return {"outputs":outs,"inputs":ins,"instrs":instrs,"vals":allv,
            "n_plan_out":len(plan.outputs)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cases",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--shard",type=int,default=0); ap.add_argument("--nshards",type=int,default=1)
    a=ap.parse_args()
    BR={}
    with open("/data/jwen929/mobile/findings/vgf/bisect_results.tsv") as f:
        hdr=f.readline().rstrip("\n").split("\t"); ix={c:i for i,c in enumerate(hdr)}
        for line in f:
            r=line.rstrip("\n").split("\t")
            if len(r)<=ix["deleg_full"]: continue
            BR[(r[0],r[1])]=(r[ix["verdict"]],r[ix["needed_siblings"]],r[ix["needed_live"]])
    cases=[l.split() for l in open(a.cases) if l.strip()]
    mine=[c for i,c in enumerate(cases) if i%a.nshards==a.shard]
    fh=open(a.out,"a",buffering=1)
    for job,oi in mine:
        rec={"job":job,"out":oi}
        try:
            v,ns,nl=BR[(job,oi)]
            needed=[x for x in (ns if v=="SIBLING_DEPENDENT" else nl).split(",") if x]
            rec["verdict"]=v
            if v=="SIBLING_DEPENDENT":
                others,bos,target,ret=output_set_localizer(C.job_file(CO,job,"py"),int(oi))
                sel=[o for o in ret if o in set([target]+needed)]
                srcA=bos([target]); srcB=bos([target]+needed); pB=sel.index(target); pA=0
                rec["sel"]=sel
            else:
                anc,build_src,target=cone_localizer(C.job_file(CO,job,"py"),int(oi))
                srcA=build_src([]); srcB=build_src(needed); pA=None; pB=None
            rec["target"]=target
            for tag,src,p in (("A",srcA,pA),("B",srcB,pB)):
                j=build_job(src,"vgf",False)
                if j.status!="READY": rec[tag]={"build":j.status}; continue
                pi=plan_info(j.pte)
                pp = p if p is not None else len(j.eager)-1
                bufs,err=run_pte(j,"d"+tag)
                d={"deleg":j.delegated_ops,"user_pos":list(j.user_pos) if j.user_pos else None,
                   "n_eager":len(j.eager),"n_plan_out":pi["n_plan_out"],"pos":pp,
                   "eager_shapes":[list(t.shape) for t in j.eager],
                   "eager_dtypes":[str(t.dtype) for t in j.eager],
                   "plan_out":pi["outputs"],"plan_in":pi["inputs"],"instrs":pi["instrs"],
                   "vals":pi["vals"],"err":err}
                if bufs is not None:
                    d["n_out_files"]=len(bufs); d["out_bytes"]=[len(b) for b in bufs]
                    e=j.eager[pp]
                    # does ANY out file reproduce eager[pp]?
                    matches=[]; zeros=[]
                    for bi,b in enumerate(bufs):
                        try:
                            t=torch.frombuffer(bytearray(b),dtype=e.dtype)
                        except Exception: matches.append(None); zeros.append(None); continue
                        zeros.append(bool(t.numel()>0 and t.float().abs().max()<1e-6))
                        if t.numel()==e.numel():
                            st,_=cmp._cmp(e,t.reshape(e.shape),0); matches.append(st)
                        else: matches.append("NUMEL")
                    d["any_out_matches_target"]=matches; d["out_is_zero"]=zeros
                    # per-eager-output full comparison using harness selection
                    ts,fb,selidx=to_tensors(bufs,j.eager,j.user_pos)
                    d["fellback"]=fb; d["selidx"]=selidx
                    per=[]
                    for k,t in enumerate(ts):
                        ee=j.eager[k]
                        if t.numel()!=ee.numel(): per.append("NUMEL"); continue
                        st,_=cmp._cmp(ee,t.reshape(ee.shape),k); per.append(st)
                    d["per_out_status"]=per
                    d["eager_absmax"]=[float(t.float().abs().max()) if t.numel() else 0.0 for t in j.eager]
                rec[tag]=d
        except Exception as e:
            import traceback; rec["error"]=traceback.format_exc()[-400:]
        fh.write(json.dumps(rec)+"\n"); print("done",job,oi,flush=True)

if __name__=="__main__": main()
