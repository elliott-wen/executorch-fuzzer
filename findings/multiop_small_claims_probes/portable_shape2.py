#!/usr/bin/env python3
"""Claim D per-case CAUSAL test (fork-isolated) -- see portable_shape.py docstring."""
import os, sys, re, warnings, logging, ast, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0,"/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from pathlib import Path
from collections import Counter
TSV="/data/jwen929/mobile/tmp/run_portable/skip_reasons_portable.tsv"
CO="/data/jwen929/mobile/corpus_v1/portable"
OUTK=("out","values","indices","min","min_indices","max","max_indices")
PAT=re.compile(r"\b("+"|".join(OUTK)+r")\s*=\s*(n\d+)\b")
def job_path(job):
    w,i=job.split(":"); return Path(CO)/w/f"{w}_{i}.py"
def run_eager(src):
    ns={}; exec(compile(src,"<g>","exec"), ns)
    r=ns["g"](*[t.clone() for t in ns["LEAVES"]])
    if isinstance(r,torch.Tensor): r=[r]
    return [tuple(x.shape) if isinstance(x,torch.Tensor) else None for x in r]
def one(job, oi, eag_sh, port_sh):
    p=job_path(job)
    if not p.exists(): return ("NO_SRC", None)
    src=p.read_text()
    if not PAT.search(src): return ("NO_ALIASED_OUT", [job,oi,str(eag_sh),str(port_sh),"-"])
    src2=PAT.sub(lambda m: f"{m.group(1)}={m.group(2)}.clone()", src)
    try: base=run_eager(src)
    except Exception as e: return ("EAGER_EXC_BASE", [job,oi,str(eag_sh),str(port_sh),type(e).__name__])
    try: dea=run_eager(src2)
    except Exception as e: return ("EAGER_EXC_DEALIAS", [job,oi,str(eag_sh),str(port_sh),type(e).__name__])
    if oi>=len(base) or oi>=len(dea): return ("IDX_OOR", [job,oi,str(eag_sh),str(port_sh),"-"])
    if base[oi]!=eag_sh: return ("BASE_DISAGREE", [job,oi,str(eag_sh),str(base[oi]),str(dea[oi])])
    if dea[oi]==port_sh: return ("CONFIRMED", None)
    if dea[oi]==eag_sh: return ("DEALIAS_NO_EFFECT", [job,oi,str(eag_sh),str(port_sh),str(dea[oi])])
    return ("DEALIAS_OTHER", [job,oi,str(eag_sh),str(port_sh),str(dea[oi])])

rows=[]
for l in open(TSV):
    c=l.rstrip("\n").split("\t")
    if len(c)>=3 and c[0]=="MISMATCH" and "shape" in c[2]:
        m=re.match(r"out\[(\d+)\] shape (\(.*?\)) vs (\(.*?\))\s*$", c[2])
        rows.append((c[1], int(m.group(1)), ast.literal_eval(m.group(2)), ast.literal_eval(m.group(3))) if m else (c[1],None,None,None))
print(f"parsed {len(rows)} shape rows; unparsed={sum(1 for r in rows if r[1] is None)}")
n_alias_present=0
for j,o,e,pp in rows:
    if o is not None and job_path(j).exists() and PAT.search(job_path(j).read_text()): n_alias_present+=1
print(f"source contains an aliased out=/values=/min=... node arg: {n_alias_present}/{len(rows)}")
ctr=Counter(); details=[]
for job, oi, eag_sh, port_sh in rows:
    if oi is None: ctr["UNPARSED"]+=1; continue
    r,w=os.pipe(); pid=os.fork()
    if pid==0:
        os.close(r)
        try: res=one(job,oi,eag_sh,port_sh)
        except BaseException as e: res=("PYEXC:"+type(e).__name__, None)
        try: os.write(w, json.dumps(res).encode())
        finally: os._exit(0)
    os.close(w)
    with os.fdopen(r,"rb") as rf: buf=rf.read()
    _,status=os.waitpid(pid,0)
    if not buf:
        ctr["NATIVE_ABORT"]+=1; details.append(("NATIVE_ABORT",[job,oi,str(eag_sh),str(port_sh),"-"])); continue
    b,d=json.loads(buf.decode())
    ctr[b]+=1
    if d: details.append((b,d))
print("RESULT", json.dumps(dict(ctr), indent=0))
print("\n-- non-CONFIRMED (bucket, [job,out,eager_sh,portable_sh,dealiased_sh]):")
for d in details[:80]: print("  ",d)
