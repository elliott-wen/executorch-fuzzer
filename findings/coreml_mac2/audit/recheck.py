import os,sys,warnings,logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0,"/data/jwen929"); warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from mobile.gen.export import build_job
from mobile.net.et_runner import run_pte
from mobile.net import compare as cmp, corpus as C
BK=os.environ["BK"]
CASES=[("w101:421",3),("w124:557",2),("w65:544",1)]
for jid,oi in CASES:
    src=open(C.job_file("/data/jwen929/mobile/corpus_v4/coreml",jid,"py")).read()
    j=build_job(src,BK)
    print(f"\n{jid} out[{oi}] backend={BK} build={j.status} user_pos={j.user_pos} n_eager={len(j.eager)}")
    if j.status!="READY": print("  ",str(j.detail)[:200]); continue
    for rep in range(3):
        raw=run_pte(j.pte,j.inputs)
        et=cmp.select(raw,j.user_pos)
        if len(et)>len(j.eager): et=et[len(et)-len(j.eager):]
        v=cmp._cmp(j.eager[oi],et[oi],0)
        e=j.eager[oi].flatten().tolist(); d=et[oi].flatten().tolist()
        allz=all(abs(x)<1e-9 for x in d)
        print(f"  rep{rep} verdict={v[0]}/{v[1] if len(v)>1 else ''} eager[:6]={[round(x,4) for x in e[:6]]} dev[:6]={[round(x,4) for x in d[:6]]} numel={len(d)} dev_all_zero={allz} dtype={j.eager[oi].dtype}")
