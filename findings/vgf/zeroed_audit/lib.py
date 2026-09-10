import os, sys, subprocess, tempfile, pathlib, ast, functools, json, shutil
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0,"/data/jwen929")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.net import corpus as C, protocol as P, compare as cmp
from mobile.gen.export import build_job
from mobile.gen.diff.cone_predicate import output_set_localizer, cone_localizer, _load, _host_values

CO="/data/jwen929/mobile/corpus_v4/vgf"
RUNNER="/data/jwen929/mobile/vgf_client/vgf_runner.sh"
SCRATCH=os.environ.get("ZA_TMP","/tmp/claude-2496112/-data-jwen929-mobile/e07c5b38-7e6e-4875-af54-f33813aae47b/scratchpad/za/run")
os.makedirs(SCRATCH, exist_ok=True)

@functools.lru_cache(maxsize=None)
def node_ops(job_id):
    mod=ast.parse(open(C.job_file(CO,job_id,"py")).read())
    g=next((s for s in mod.body if isinstance(s,ast.FunctionDef) and s.name=="g"),None); out={}
    for stmt in (g.body if g else []):
        if isinstance(stmt,ast.Assign) and isinstance(stmt.targets[0],ast.Name):
            op="?"
            for n in ast.walk(stmt.value):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                    parts=[]; cur=n.func
                    while isinstance(cur,ast.Attribute): parts.append(cur.attr); cur=cur.value
                    parts=parts[::-1]
                    if "aten" in parts: i=parts.index("aten"); op=parts[i+1] if i+1<len(parts) else "?"; break
            out[stmt.targets[0].id]=op
    return out

def run_pte(j, tag):
    """Run job j on the vgf runtime; return (list_of_raw_bytes, err)."""
    d=pathlib.Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=SCRATCH))
    pte=d/"m.pte"; pte.write_bytes(j.pte)
    cmd=[RUNNER,"--pte",str(pte),"--out",str(d)]
    for i,t in enumerate(j.inputs):
        _,raw=P.tensor_to_meta_blob(t.contiguous())
        p=d/f"in_{i}.bin"; p.write_bytes(raw); cmd+=["--input",str(p)]
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=900)
    except subprocess.TimeoutExpired:
        shutil.rmtree(d,ignore_errors=True); return None,"TIMEOUT"
    if r.returncode!=0:
        shutil.rmtree(d,ignore_errors=True); return None,f"rc={r.returncode}"
    fs=sorted(d.glob("out_*.bin"), key=lambda p:int(p.stem.split("_")[1]))
    bufs=[f.read_bytes() for f in fs]
    shutil.rmtree(d,ignore_errors=True)
    return bufs,None

def to_tensors(bufs, eager, user_pos):
    """Replicate the harness path: select(user_pos) then trailing-trim, then reshape by eager."""
    sel_idx=list(range(len(bufs)))
    fellback=False
    if user_pos is not None and all(0<=i<len(bufs) for i in user_pos):
        sel_idx=list(user_pos)
    else:
        fellback=True
    if len(sel_idx)>len(eager): sel_idx=sel_idx[len(sel_idx)-len(eager):]
    ts=[]
    for k,bi in enumerate(sel_idx):
        e=eager[k] if k<len(eager) else eager[-1]
        t=(torch.empty(0,dtype=e.dtype) if len(bufs[bi])==0 else torch.frombuffer(bytearray(bufs[bi]),dtype=e.dtype))
        ts.append(t)
    return ts, fellback, sel_idx

def mech(ref,dev):
    r=ref.flatten().float(); d=dev.flatten().float()
    if r.numel()!=d.numel(): return "SHAPE"
    rf=torch.isfinite(r); df=torch.isfinite(d)
    if (rf!=df).any(): return "NONFINITE"
    if d.abs().max()<1e-6 and r.abs().max()>1e-3: return "ZEROED"
    return "VALUE"

def verdict_at(eager, ts, pos):
    e=eager[pos]
    d=ts[pos]
    if d.numel()!=e.numel(): return "SHAPE", f"dev_numel={d.numel()} vs {e.numel()}", None
    d=d.reshape(e.shape)
    st,det=cmp._cmp(e,d,pos)
    if st!="MISMATCH": return "OK",det,d
    return "MISMATCH/"+mech(e,d), det, d
