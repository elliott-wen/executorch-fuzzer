#!/usr/bin/env python3
"""nf_vd.py — build-only: VGF delegated_ops for the B-side graph of every NONFINITE row.
A row with delegated_ops==0 has NO VGF delegate in the graph that diverged, so the
divergence cannot be a VGF-backend finding at all."""
import os, sys, json, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from nf_common import load_rows, srcs_for
from mobile.gen.export import build_job
ap = argparse.ArgumentParser(); ap.add_argument("out")
ap.add_argument("--mech", default="NONFINITE")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
a = ap.parse_args()
rows = [r for r in load_rows() if r["mechanism"] == a.mech]
rows.sort(key=lambda r: (r["job"], r["out"]))
mine = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: d = json.loads(l); done.add((d["job"], d["out"]))
        except Exception: pass
fh = open(a.out, "a", buffering=1)
for r in mine:
    if (r["job"], r["out"]) in done: continue
    rec = dict(job=r["job"], out=r["out"], verdict=r["verdict"], tgt_op=r["tgt_op"])
    try:
        sA, sB, t, pA, pB = srcs_for(r["job"], r["out"], r["verdict"], r["needed"])
        jB = build_job(sB, "vgf", False)
        rec.update(status=jB.status, d=getattr(jB, "delegated_ops", -1),
                   nd=getattr(jB, "non_delegated_ops", -1), calls=getattr(jB, "delegate_calls", -1),
                   dtype=str(jB.eager[pB if pB >= 0 else -1].dtype).replace("torch.", "") if jB.status == "READY" else "?")
    except BaseException as e:
        rec["status"] = f"EXC:{type(e).__name__}:{str(e)[:60]}"
    fh.write(json.dumps(rec) + "\n")
    print(f"[vd s{a.shard}] {r['job']}:{r['out']} {rec.get('status')} d={rec.get('d')} nd={rec.get('nd')}", flush=True)
print("DONE", a.shard, flush=True)
