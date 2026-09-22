"""Host-only: for every graphopt_mechanisms row, get the EAGER reference tensor at that output
index (no lowering needed) -> dtype, numel, and whether the reference is constant.
Tests the classifier's blind spots directly."""
import os,sys,warnings,logging,collections
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); os.environ.setdefault("OMP_NUM_THREADS","2")
sys.path.insert(0,"/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.gen.export.job import load_functional_source, _flatten_outputs
from mobile.executor import corpus as C
CORPUS="/data/jwen929/mobile/corpus_v4/coreml"
D="/data/jwen929/mobile/findings/coreml_mac2"
rows=[l.split("\t") for l in open(f"{D}/graphopt_mechanisms.tsv").read().splitlines()[1:] if l.strip()]
agg=collections.defaultdict(collections.Counter)
det=collections.defaultdict(list)
for r in rows:
    jid,oi,verdict,req,mech=(r+[""]*5)[:5]
    if mech.startswith("SCALE"): key="SCALE"
    elif mech in ("ZEROED","WRONG-VALUE","NONFINITE","NEAR-TOL (gate N>=5)"): key=mech
    else: continue
    oi=int(oi)
    try:
        g,leaves=load_functional_source(open(C.job_file(CORPUS,jid,"py")).read())
        with torch.no_grad(): eg=_flatten_outputs(g(*[t.clone() for t in leaves]))
    except Exception as e:
        agg[key]["EAGER_FAIL"]+=1; continue
    if oi>=len(eg): agg[key]["OOR"]+=1; continue
    t=eg[oi]; n=t.numel()
    dt=str(t.dtype).replace("torch.","")
    isb = t.dtype==torch.bool
    isint = (not t.dtype.is_floating_point) and not isb
    const = (n>0 and bool(torch.all(t.reshape(-1)==t.reshape(-1)[0]).item()))
    agg[key]["TOTAL"]+=1
    agg[key][f"dtype={dt}"]+=1
    if n==1: agg[key]["numel==1"]+=1
    if const: agg[key]["const_reference"]+=1
    if isb: agg[key]["BOOL_output"]+=1
    if isint: agg[key]["INT_output"]+=1
    if n==1 or const: agg[key]["numel1_OR_const"]+=1
    det[key].append((jid,oi,dt,n,const))
for k in agg:
    print(f"\n=== {k} ===")
    for kk,v in sorted(agg[k].items(), key=lambda x:-x[1]): print(f"  {v:5d}  {kk}")
