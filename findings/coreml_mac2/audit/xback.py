"""Cross-backend control at CORPUS scale: run the FULL graph of every ZEROED case on
portable / xnnpack in-process and report the verdict at the SAME output index.
If portable/xnnpack also ZERO that output, the CoreML-specific attribution fails for that case."""
import os,sys,warnings,logging,collections,argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); os.environ.setdefault("OMP_NUM_THREADS","2")
sys.path.insert(0,"/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from mobile.gen.export import build_job
from mobile.net.et_runner import run_pte
from mobile.net import compare as cmp, corpus as C
CORPUS="/data/jwen929/mobile/corpus_v4/coreml"
D="/data/jwen929/mobile/findings/coreml_mac2"
ap=argparse.ArgumentParser(); ap.add_argument("--mech",default="ZEROED"); ap.add_argument("--bk",default="portable")
a=ap.parse_args()
work=[(r[0],int(r[1])) for r in (l.split("\t") for l in open(f"{D}/graphopt_mechanisms.tsv").read().splitlines()[1:] if l.strip()) if r[4]==a.mech]
cnt=collections.Counter()
for jid,oi in work:
    src=open(C.job_file(CORPUS,jid,"py")).read()
    try: j=build_job(src,a.bk)
    except Exception as e: print(f"{jid}\t{oi}\tBUILD_EXC\t{type(e).__name__}"); cnt["BUILD_EXC"]+=1; continue
    if j.status!="READY": print(f"{jid}\t{oi}\tBUILD:{j.status}\t{str(j.detail)[:60]}"); cnt["BUILD:"+j.status]+=1; continue
    try: raw=run_pte(j.pte,j.inputs)
    except Exception as e: print(f"{jid}\t{oi}\tRUN_EXC\t{type(e).__name__}"); cnt["RUN_EXC"]+=1; continue
    et=cmp.select(raw,j.user_pos)
    if len(et)>len(j.eager): et=et[len(et)-len(j.eager):]
    if oi>=len(et): print(f"{jid}\t{oi}\tNO-OUT\t(n={len(et)})"); cnt["NO-OUT"]+=1; continue
    v=cmp._cmp(j.eager[oi],et[oi],0)[0]
    dl=et[oi].flatten().tolist(); el=j.eager[oi].flatten().tolist()
    z = len(dl)>0 and all(abs(x)<1e-9 for x in dl if isinstance(x,float) or isinstance(x,int)) and any(abs(x)>1e-6 for x in el)
    tag=v+("+ZEROED" if z else "")
    cnt[tag]+=1
    print(f"{jid}\t{oi}\t{tag}\teager={[round(x,4) for x in el[:4]]}\tdev={[round(x,4) for x in dl[:4]]}")
print(f"\n=== SUMMARY {a.bk} mech={a.mech} n={len(work)} ===")
for k,v in cnt.most_common(): print(f"  {v:4d}  {k}")
