# control: sample jobs that were OK on device, check how often ANY user output carries the
# DUP / PASSTHRU AOT signature. If OK jobs carry it often, the signature is not predictive.
import os,sys,random,collections,warnings,logging,glob,re
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); os.environ.setdefault("OMP_NUM_THREADS","2")
sys.path.insert(0,"/data/jwen929")
sys.path.insert(0,"/tmp/claude-2496112/-data-jwen929-mobile/e07c5b38-7e6e-4875-af54-f33813aae47b/scratchpad/audit")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from mobile.net import corpus as C
D="/data/jwen929/mobile/findings/coreml_mac2"
nonok={l.split("\t")[1] for l in open(f"{D}/skiplog.tsv").read().splitlines()[1:] if l.strip()}
allj=[]
for p in glob.glob("/data/jwen929/mobile/corpus_v4/coreml/w*/w*_*.py"):
    m=re.search(r"/(w\d+)/w\d+_(\d+)\.py$",p)
    if m: allj.append(f"{m.group(1)}:{m.group(2)}")
ok=[j for j in allj if j not in nonok]
random.seed(7); samp=random.sample(ok,int(os.environ.get("N","150")))
print("total corpus py:",len(allj),"OK-eligible:",len(ok),"sampled:",len(samp))
import importlib.util
spec=importlib.util.spec_from_file_location("sg","/tmp/claude-2496112/-data-jwen929-mobile/e07c5b38-7e6e-4875-af54-f33813aae47b/scratchpad/audit/sig_scan.py")
# can't import (it runs main). inline the analyze instead:
import json
import torch
from torch.export import export
from torch.export.graph_signature import OutputKind
from mobile.gen.export.job import _GraphModule, load_functional_source
from mobile.gen.export.backends import get_backend
BE=get_backend("coreml")
def hdr(b):
    b=bytes(b); depth=0
    for i,c in enumerate(b[:8192]):
        if c==0x7b: depth+=1
        elif c==0x7d:
            depth-=1
            if depth==0: return json.loads(b[:i+1].decode())
def analyze(src):
    g,leaves=load_functional_source(src)
    ep=export(_GraphModule(g).eval(),tuple(t.clone() for t in leaves))
    exe=BE.lower(ep,tuple(t.clone() for t in leaves),quantize=False)
    p=exe.exported_program(); gm=p.graph_module
    heads={n.name:hdr(getattr(gm,n.target).processed_bytes) for n in gm.graph.nodes if n.op=="get_attr"}
    src_of={}
    for n in gm.graph.nodes:
        if n.op=="call_function" and "getitem" in str(n.target):
            par=n.args[0]
            if getattr(par,"op",None)=="call_function" and "call_delegate" in str(par.target):
                src_of[n.name]=(par.args[0].name,int(n.args[1]))
    specs=p.graph_signature.output_specs
    flat=list(gm.graph.nodes)[-1].args[0]
    if not isinstance(flat,(tuple,list)): flat=(flat,)
    out=[]
    for i,s in enumerate(specs):
        if s.kind!=OutputKind.USER_OUTPUT: continue
        nm=getattr(flat[i],"name",None) if i<len(flat) else None
        tag="NOT-DELEGATED"
        if nm in src_of:
            ga,oi=src_of[nm]; h=heads.get(ga)
            if h:
                ons=h.get("outputNames",[]); ins=h.get("inputNames",[])
                on=ons[oi] if oi<len(ons) else None
                tag="DUP" if (on and ons.count(on)>1) else ("PASSTHRU" if (on and on in ins) else "NEITHER")
        out.append(tag)
    return out
c=collections.Counter(); pj=collections.Counter(); n_ok=0
for j in samp:
    try: tags=analyze(open(C.job_file("/data/jwen929/mobile/corpus_v4/coreml",j,"py")).read())
    except Exception as e: c["ERR"]+=1; continue
    n_ok+=1
    for t in tags: c[t]+=1
    hit = "DUP" if "DUP" in tags else ("PASSTHRU" if "PASSTHRU" in tags else "none")
    pj[hit]+=1
print("\n=== OK-CONTROL: per-OUTPUT tags over",n_ok,"device-OK jobs ===")
for k,v in c.most_common(): print(f"  {v:5d}  {k}")
print("=== per-JOB: does the job have ANY DUP/PASSTHRU output? ===")
for k,v in pj.most_common(): print(f"  {v:5d}  {k}")
