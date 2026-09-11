import os,sys,re,warnings,logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES","");os.environ.setdefault("MOBILE_BACKENDS","ethos-u")
sys.path.insert(0,"/data/jwen929"); warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.gen.export import build_job
from mobile.executor import corpus as C
CO="/data/jwen929/mobile/corpus_v2/ethos-u"
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from eu_probe import parse, cone, mksrc
DEV=[-1.1619468927383423,-0.16088494658470154,0.0]
for job,ti,sib in [("w78:202",4,"n6"),("w87:268",2,"n7")]:
    PRE,nl,order,ins,ret,POST=parse(job); tgt=ret[ti]
    for label,outs in [("A",[tgt]),("B",[tgt,sib])]:
        src=mksrc(PRE,nl,order,ins,POST,outs)
        j=build_job(src,"ethos-u",True)
        if j.status!="READY": print(job,label,j.status); continue
        # re-export to find quant params: rebuild the PT2E-converted module
        print(f"\n{job} {label}: eager={[float(x) for x in j.eager[0].flatten()[:4].tolist()]}")
        for k,t in enumerate(j.inputs):
            print(f"   input[{k}] {tuple(t.shape)} {t.dtype} {[float(x) for x in t.flatten()[:4].tolist()]}")
        # candidate scale from DEV: DEV = q*s
        for s in [abs(DEV[0])/n for n in range(1,128)]:
            qs=[DEV[0]/s, DEV[1]/s]
            if all(abs(q-round(q))<1e-6 for q in qs) and abs(round(qs[1]))>0:
                print(f"   DEV decomposes as q*s: s={s!r} q={[round(q) for q in qs]}")
                break
