#!/usr/bin/env python3
"""nf_anc.py — the COMPOSITIONAL confound test.

For a COMPOSITIONAL row, side A bakes the target's ancestors as fp32 CONSTANTS while side B
runs them LIVE on the device. So B feeds the target a *different input* than A does. If the
live ancestor value already diverges from eager, the target's non-finite output is ordinary
error propagation into a singular function (log/div/rsqrt/acos/...), NOT the compiler treating
the target differently in context — i.e. not a graph-optimization defect.

Test: emit the live ancestors alongside the target and compare THEM to eager.
"""
import os, sys, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nf_common import *
ap = argparse.ArgumentParser(); ap.add_argument("out")
ap.add_argument("--only", default="")   # comma list of job:out to restrict to
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
a = ap.parse_args()
keep = set(a.only.split(",")) if a.only else None
rows = [r for r in load_rows() if r["mechanism"] == "NONFINITE" and r["verdict"] == "COMPOSITIONAL"]
rows.sort(key=lambda r: (r["job"], r["out"]))
if keep: rows = [r for r in rows if f"{r['job']}:{r['out']}" in keep]
mine = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: d = json.loads(l); done.add((d["job"], d["out"]))
        except Exception: pass
fh = open(a.out, "a", buffering=1)
for r in mine:
    if (r["job"], r["out"]) in done: continue
    rec = dict(job=r["job"], out=r["out"], tgt_op=r["tgt_op"], needed=r["needed"])
    try:
        sA, sB, t, pA, pB = srcs_for(r["job"], r["out"], r["verdict"], r["needed"])
        live = [n for n in r["needed"] if re.search(rf"^    {re.escape(n)} = ", sB, re.M)]
        rec["live_emitted"] = live
        if not live:
            rec["res"] = "NO_LIVE_ANCESTOR"; fh.write(json.dumps(rec) + "\n"); continue
        src2 = re.sub(r"    return \((\w+),\)\s*$",
                      lambda m: f"    return ({m.group(1)}, " + ", ".join(live) + ",)\n", sB)
        if src2 == sB: rec["res"] = "REWRITE_FAIL"; fh.write(json.dumps(rec) + "\n"); continue
        j = build_job(src2, "vgf", False)
        rec["status"] = j.status
        if j.status == "READY":
            raws, err = vgf_run(j, "anc")
            if raws is None: rec["res"] = f"RUN:{err[:50]}"
            else:
                al = align(raws, j)
                if al is None:
                    rec["res"] = f"ALIGN_FAIL nraw={len(raws)} neager={len(j.eager)} up={j.user_pos}"
                else:
                    # outputs: [target] + live, in emitted order
                    names = ["TARGET"] + live
                    per = []
                    for i, nm in enumerate(names):
                        if i >= len(al): break
                        st, m, k = classify(j.eager[i], al[i])
                        per.append(dict(name=nm, res=st, mech=m, kind=k,
                                        ref_nf=nf_signature(j.eager[i]), dev_nf=nf_signature(al[i])))
                    rec["per_output"] = per
                    anc = [p for p in per if p["name"] != "TARGET"]
                    rec["res"] = "OK"
                    rec["target_res"] = per[0]["res"] if per else "?"
                    rec["ancestor_diverges"] = any(p["res"] == "MISMATCH" for p in anc)
                    rec["n_anc_div"] = sum(1 for p in anc if p["res"] == "MISMATCH")
                    rec["n_anc"] = len(anc)
    except BaseException as e:
        rec["res"] = f"EXC:{type(e).__name__}:{str(e)[:80]}"
    fh.write(json.dumps(rec) + "\n")
    print(f"[anc s{a.shard}] {r['job']}:{r['out']} {r['tgt_op']} {rec.get('res')} "
          f"tgt={rec.get('target_res')} anc_div={rec.get('n_anc_div')}/{rec.get('n_anc')}", flush=True)
print("DONE", a.shard, flush=True)
