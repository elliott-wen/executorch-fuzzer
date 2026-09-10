import os,sys,warnings,logging,re,collections
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0,"/data/jwen929"); warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.gen.export.job import load_functional_source,_flatten_outputs
from mobile.net import corpus as C
CORPUS="/data/jwen929/mobile/corpus_v4/coreml"; D="/data/jwen929/mobile/findings/coreml_mac2"
rows=[l.split("\t") for l in open(f"{D}/graphopt_mechanisms.tsv").read().splitlines()[1:] if l.strip()]
sc=[r for r in rows if r[4].startswith("SCALE")]
def nm1n(r,tol=0.012):
    for N in range(2,129):
        if abs(r-(N-1)/N)<=tol*max(1e-9,abs(r)): return N
    return None
print(f"{'job':11s} {'oi':>2s} {'ratio':>10s} {'(N-1)/N?':>9s} {'numel':>5s} {'const':>5s} {'dtype':8s} {'var?':4s} POWER")
agg=collections.Counter()
for r in sc:
    jid,oi=r[0],int(r[1]); mech=r[4]
    ratio=float(mech.split(":")[1])
    txt=open(C.job_file(CORPUS,jid,"py")).read()
    hasvar=bool(re.search(r"\b(var|var_mean|std)\.correction",txt))
    try:
        g,l=load_functional_source(txt)
        with torch.no_grad(): eg=_flatten_outputs(g(*[t.clone() for t in l]))
        t=eg[oi]; n=t.numel(); const=bool(torch.all(t.reshape(-1)==t.reshape(-1)[0]).item()) if n else True
        dt=str(t.dtype).replace("torch.","")
    except Exception: n,const,dt=-1,True,"?"
    N=nm1n(ratio)
    power = "VACUOUS" if (n==1 or const) else "REAL"
    agg[(power,"var" if hasvar else "novar")]+=1
    agg[("Nform" if N else "notNform","var" if hasvar else "novar")]+=1
    print(f"{jid:11s} {oi:2d} {ratio:10.4g} {str(N or '-'):>9s} {n:5d} {str(const):>5s} {dt:8s} {'VAR' if hasvar else '-':4s} {power}")
print("\n=== crosstabs (n=%d) ==="%len(sc))
for k,v in sorted(agg.items()): print(f"  {v:4d}  {k}")
print("\nSCALE with real discriminating power (numel>1 AND non-constant ref):", sum(v for k,v in agg.items() if k[0]=='REAL'))
print("SCALE whose ratio matches (N-1)/N within 1.2%:", sum(v for k,v in agg.items() if k[0]=='Nform'))
print("graphs containing var/std.correction:", sum(v for k,v in agg.items() if k[0] in ('REAL','VACUOUS') and k[1]=='var'))
