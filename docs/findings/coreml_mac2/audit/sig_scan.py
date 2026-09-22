import os,sys,warnings,logging,json,argparse,collections,traceback
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); os.environ.setdefault("OMP_NUM_THREADS","2")
sys.path.insert(0,"/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from torch.export import export
from torch.export.graph_signature import OutputKind
from mobile.gen.export.job import _GraphModule, load_functional_source
from mobile.gen.export.backends import get_backend
from mobile.executor import corpus as C
CORPUS="/data/jwen929/mobile/corpus_v4/coreml"
BE=get_backend("coreml")

def hdr(b):
    b=bytes(b); depth=0
    for i,c in enumerate(b[:8192]):
        if c==0x7b: depth+=1
        elif c==0x7d:
            depth-=1
            if depth==0: return json.loads(b[:i+1].decode())
    return None

def analyze(src):
    g,leaves=load_functional_source(src)
    ep=export(_GraphModule(g).eval(),tuple(t.clone() for t in leaves))
    exe=BE.lower(ep,tuple(t.clone() for t in leaves),quantize=False)
    p=exe.exported_program(); gm=p.graph_module
    heads={}
    for n in gm.graph.nodes:
        if n.op=="get_attr":
            try: heads[n.name]=hdr(getattr(gm,n.target).processed_bytes)
            except Exception: pass
    # node -> (delegate_attr_name, out_index)
    src_of={}
    for n in gm.graph.nodes:
        if n.op=="call_function" and "getitem" in str(n.target):
            par=n.args[0]
            if getattr(par,"op",None)=="call_function" and "call_delegate" in str(par.target):
                ga=par.args[0]
                src_of[n.name]=(ga.name if hasattr(ga,"name") else str(ga), int(n.args[1]))
        if n.op=="call_function" and "call_delegate" in str(n.target):
            pass
    specs=p.graph_signature.output_specs
    outnode=list(gm.graph.nodes)[-1]
    flat=outnode.args[0]
    if not isinstance(flat,(tuple,list)): flat=(flat,)
    res=[]
    ui=0
    for i,s in enumerate(specs):
        if s.kind!=OutputKind.USER_OUTPUT: continue
        nd=flat[i] if i<len(flat) else None
        nm=getattr(nd,"name",None)
        info={"user_idx":ui,"node":nm,"deleg":None,"outname":None,"DUP":False,"PASSTHRU":False}
        if nm in src_of:
            ga,oi=src_of[nm]; h=heads.get(ga)
            if h:
                ons=h.get("outputNames",[]); ins=h.get("inputNames",[])
                on=ons[oi] if oi<len(ons) else None
                info.update(deleg=ga,outname=on,
                            DUP=(on is not None and ons.count(on)>1),
                            PASSTHRU=(on is not None and on in ins),
                            all_out=ons, all_in=ins)
        res.append(info); ui+=1
    return res

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mech",default="ZEROED"); ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--jobs",default=""); a=ap.parse_args()
    D="/data/jwen929/mobile/findings/coreml_mac2"
    if a.jobs:
        work=[(j.split("@")[0],int(j.split("@")[1])) for j in a.jobs.split(",")]
    else:
        work=[(r[0],int(r[1])) for r in (l.split("\t") for l in open(f"{D}/graphopt_mechanisms.tsv").read().splitlines()[1:] if l.strip()) if r[4]==a.mech or (a.mech=="SCALE" and r[4].startswith("SCALE"))]
    if a.limit: work=work[:a.limit]
    cnt=collections.Counter()
    for jid,oi in work:
        try:
            res=analyze(open(C.job_file(CORPUS,jid,"py")).read())
        except Exception as e:
            print(f"{jid}\t{oi}\tERR\t{type(e).__name__}: {str(e)[:70]}"); cnt["ERR"]+=1; continue
        m=next((r for r in res if r["user_idx"]==oi),None)
        if m is None:
            print(f"{jid}\t{oi}\tNO-OUT\t(n_user={len(res)})"); cnt["NO-OUT"]+=1; continue
        tag="DUP" if m["DUP"] else ("PASSTHRU" if m["PASSTHRU"] else ("NEITHER" if m["outname"] else "NOT-DELEGATED"))
        cnt[tag]+=1
        print(f"{jid}\t{oi}\t{tag}\toutname={m['outname']}\tdeleg={m['deleg']}\tnode={m['node']}")
    print("\n=== SUMMARY mech="+a.mech+f" n={len(work)} ===")
    for k,v in cnt.most_common(): print(f"  {v:4d}  {k}")
main()
