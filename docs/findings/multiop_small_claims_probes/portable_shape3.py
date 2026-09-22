#!/usr/bin/env python3
"""Claim D, test 2: is the EAGER divergent shape independent of the DECLARED
out= buffer shape? Perturb every INLINE `out=torch.empty((..), dtype=D)` (and
values=/indices=/min=/max=) to a different arbitrary shape and re-run eager.
  INVARIANT  -> eager returns the op's TRUE shape regardless of the buffer, i.e.
                eager resizes; portable returns the DECLARED buffer shape.
                => divergence fully explained by wrong-shaped generator buffers.
  CHANGED    -> the declared shape does leak into the eager result.
Also records whether the PORTABLE-reported shape equals the divergent node's own
declared inline out= buffer shape."""
import os, sys, re, warnings, logging, ast, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0,"/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from pathlib import Path
from collections import Counter
TSV="/data/jwen929/mobile/tmp/run_portable/skip_reasons_portable.tsv"
CO="/data/jwen929/mobile/corpus_v1/portable"
KW=("out","values","indices","min","min_indices","max","max_indices")
INLINE=re.compile(r"\b("+"|".join(KW)+r")\s*=\s*torch\.empty\(\((.*?)\)\s*,\s*dtype=([A-Za-z0-9_.]+)\)")
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
    if not INLINE.search(src): return ("NO_INLINE_OUT", [job,oi,str(eag_sh),str(port_sh)])
    # perturb every inline out= buffer shape to something unrelated
    src2=INLINE.sub(lambda m: f"{m.group(1)}=torch.empty((3, 5), dtype={m.group(3)})", src)
    try: base=run_eager(src)
    except Exception as e: return ("EAGER_EXC_BASE",[job,oi,type(e).__name__])
    try: pert=run_eager(src2)
    except Exception as e: return ("EAGER_EXC_PERTURB",[job,oi,type(e).__name__])
    if oi>=len(base) or oi>=len(pert): return ("IDX_OOR",[job,oi,"-","-"])
    if base[oi]!=eag_sh: return ("BASE_DISAGREE",[job,oi,str(eag_sh),str(base[oi])])
    if pert[oi]==eag_sh: return ("EAGER_SHAPE_INVARIANT", None)
    return ("EAGER_SHAPE_CHANGED",[job,oi,str(eag_sh),str(port_sh),str(pert[oi])])
rows=[]
for l in open(TSV):
    c=l.rstrip("\n").split("\t")
    if len(c)>=3 and c[0]=="MISMATCH" and "shape" in c[2]:
        m=re.match(r"out\[(\d+)\] shape (\(.*?\)) vs (\(.*?\))\s*$", c[2])
        if m: rows.append((c[1], int(m.group(1)), ast.literal_eval(m.group(2)), ast.literal_eval(m.group(3))))
ctr=Counter(); det=[]
for job, oi, e, pp in rows:
    r,w=os.pipe(); pid=os.fork()
    if pid==0:
        os.close(r)
        try: res=one(job,oi,e,pp)
        except BaseException as ex: res=("PYEXC:"+type(ex).__name__,None)
        try: os.write(w, json.dumps(res).encode())
        finally: os._exit(0)
    os.close(w)
    with os.fdopen(r,"rb") as rf: buf=rf.read()
    os.waitpid(pid,0)
    if not buf: ctr["NATIVE_ABORT"]+=1; continue
    b,d=json.loads(buf.decode()); ctr[b]+=1
    if d: det.append((b,d))
print("RESULT", json.dumps(dict(ctr)))
for d in det[:25]: print("  ",d)
