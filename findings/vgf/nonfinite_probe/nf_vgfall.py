#!/usr/bin/env python3
"""nf_vgfall.py — B-side VGF device values for EVERY row of a mechanism bucket (1 rep).
Records the full non-finite signature AND whether the device output is all-zero, so the
NONFINITE label can be re-adjudicated against the ZEROED / ALIAS / VALUE alternatives that
mech()'s isfinite-first ordering pre-empts."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nf_common import *
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
        j = build_job(sB, "vgf", False)
        rec["status"] = j.status
        if j.status == "READY":
            p = pB if pB >= 0 else len(j.eager) - 1
            ref = j.eager[p]
            rec.update(d=j.delegated_ops, dtype=str(ref.dtype).replace("torch.", ""),
                       ref_nf=nf_signature(ref))
            raws, err = vgf_run(j, "va")
            if raws is None: rec["res"] = f"RUN:{err[:50]}"
            else:
                al = align(raws, j)
                if al is None: rec["res"] = "ALIGN_FAIL"
                else:
                    dev = al[p]
                    st, m, k = classify(ref, dev)
                    df = dev.flatten().float()
                    rec.update(res=st, mech=m, kind=k, dev_nf=nf_signature(dev),
                               dev_allzero=bool(float(df.abs().max()) == 0.0),
                               dev_absmax=float(df[df.isfinite()].abs().max()) if bool(df.isfinite().any()) else None,
                               ref_absmax=float(ref.flatten().float()[ref.flatten().float().isfinite()].abs().max()) if bool(ref.flatten().float().isfinite().any()) else None,
                               dev_sat65504=bool((df.abs() == 65504.0).any()))
    except BaseException as e:
        rec["res"] = f"EXC:{type(e).__name__}:{str(e)[:80]}"
    fh.write(json.dumps(rec) + "\n")
    print(f"[va s{a.shard}] {r['job']}:{r['out']} {r['tgt_op']} {rec.get('res')} d={rec.get('d')} "
          f"kind={rec.get('kind')} allzero={rec.get('dev_allzero')}", flush=True)
print("DONE", a.shard, flush=True)
