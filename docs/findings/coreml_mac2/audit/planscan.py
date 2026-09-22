"""For every ZEROED case: lower to coreml on this host, deserialize the .pte, and test whether the
ExecuTorch memory plan gives two USER_OUTPUTs OVERLAPPING byte ranges in the same memory_id.
If it never does, the 'aliasing happens inside the compiled CoreML model' claim survives at the
ExecuTorch level; if it does, the fault is (also) in ExecuTorch's planner."""
import os,sys,warnings,logging,collections
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); os.environ.setdefault("OMP_NUM_THREADS","2")
sys.path.insert(0,"/data/jwen929"); warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from mobile.gen.export import build_job
from mobile.executor import corpus as C
from executorch.exir._serialize._program import deserialize_pte_binary
D="/data/jwen929/mobile/findings/coreml_mac2"
SZ={0:1,1:1,2:2,3:4,4:8,5:2,6:4,7:8,11:1,15:2,16:4,17:8}
work=[(r[0],int(r[1])) for r in (l.split("\t") for l in open(f"{D}/graphopt_mechanisms.tsv").read().splitlines()[1:] if l.strip()) if r[4]=="ZEROED"]
cnt=collections.Counter()
for jid,oi in work:
    try: j=build_job(open(C.job_file("/data/jwen929/mobile/corpus_v4/coreml",jid,"py")).read(),"coreml")
    except Exception as e: cnt["BUILD_EXC"]+=1; print(f"{jid}\t{oi}\tBUILD_EXC"); continue
    if j.status!="READY": cnt["BUILD:"+j.status]+=1; print(f"{jid}\t{oi}\tBUILD:{j.status}"); continue
    try:
        prog=deserialize_pte_binary(j.pte); prog=getattr(prog,"program",prog); plan=prog.execution_plan[0]
        segs=[]
        for k,vi in enumerate(plan.outputs):
            if j.user_pos and k not in j.user_pos: continue
            v=plan.values[vi].val
            if type(v).__name__!="Tensor": continue
            ai=v.allocation_info
            if ai is None: segs.append((k,None,None,None)); continue
            n=1
            for s in v.sizes: n*=s
            nb=n*SZ.get(v.scalar_type,4)
            segs.append((k,ai.memory_id,ai.memory_offset_low,nb))
        ov=[]
        for i in range(len(segs)):
            for m in range(i+1,len(segs)):
                a,b=segs[i],segs[m]
                if a[1] is None or b[1] is None or a[1]!=b[1]: continue
                if a[2]<b[2]+b[3] and b[2]<a[2]+a[3]: ov.append((a[0],b[0]))
        tag="PLAN_OVERLAP" if ov else "PLAN_DISJOINT"
        cnt[tag]+=1
        print(f"{jid}\t{oi}\t{tag}\t{ov if ov else ''}\tsegs={segs}")
    except Exception as e:
        cnt["PLAN_EXC"]+=1; print(f"{jid}\t{oi}\tPLAN_EXC\t{type(e).__name__}")
print(f"\n=== SUMMARY n={len(work)} ===")
for k,v in cnt.most_common(): print(f"  {v:4d}  {k}")
