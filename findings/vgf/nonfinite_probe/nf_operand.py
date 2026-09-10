#!/usr/bin/env python3
"""nf_operand.py — is the SURVIVING NONFINITE case really the ZEROED bug one op upstream?

The survivor set is dominated by log/log2/log10/reciprocal/rsqrt/fmod/group_norm, i.e. functions
that map 0 to a non-finite value (log(0)=-inf, 1/0=+inf, x%0=nan, 0/0=nan). If the target's own
OPERAND is all-zero (or diverges) on the device, the non-finite output is the ZEROED dropped-store
defect observed through a singular function, not a separate 'non-finite' mechanism.

Test: emit the target's direct operands alongside the target and compare THEM to eager.
"""
import os, sys, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nf_common import *
from mobile.gen.diff.cone_predicate import _load
import torch
ap = argparse.ArgumentParser(); ap.add_argument("out")
ap.add_argument("--only", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
a = ap.parse_args()
keep = set(a.only.split(","))
rows = [r for r in load_rows() if r["mechanism"] == "NONFINITE" and f"{r['job']}:{r['out']}" in keep]
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
    rec = dict(job=r["job"], out=r["out"], tgt_op=r["tgt_op"], verdict=r["verdict"])
    try:
        py = C.job_file(CO, r["job"], "py")
        pre, leaves, node_val, order, deps, ret = _load(py)
        sA, sB, target, pA, pB = srcs_for(r["job"], r["out"], r["verdict"], r["needed"])
        rec["target"] = target
        # direct operands of the target that are NODES emitted in this sub-graph
        ops = [d for d in sorted(deps.get(target, ())) if re.search(rf"^    {re.escape(d)} = ", sB, re.M)]
        rec["operands"] = ops
        if not ops:
            rec["res"] = "OPERANDS_ALL_BAKED_OR_LEAF"; fh.write(json.dumps(rec) + "\n"); continue
        m = re.search(r"    return \(([^)]*)\)\s*$", sB, re.M)
        if not m: rec["res"] = "NO_RETURN"; fh.write(json.dumps(rec) + "\n"); continue
        inner = m.group(1).rstrip().rstrip(",")
        n_before = len([x for x in inner.split(",") if x.strip()])
        src2 = sB[:m.start()] + f"    return ({inner}, " + ", ".join(ops) + ",)\n"
        j = build_job(src2, "vgf", False)
        rec["status"] = j.status
        if j.status == "READY":
            raws, err = vgf_run(j, "opd")
            if raws is None: rec["res"] = f"RUN:{err[:50]}"
            else:
                al = align(raws, j)
                if al is None: rec["res"] = "ALIGN_FAIL"
                else:
                    per = []
                    for i, nm in enumerate(ops):
                        idx = n_before + i
                        if idx >= len(al): break
                        ref, dev = j.eager[idx], al[idx]
                        df = dev.flatten().float()
                        st, mm, kk = classify(ref, dev)
                        per.append(dict(name=nm, res=st, mech=mm, kind=kk,
                                        dev_allzero=bool(float(df.abs().max()) == 0.0),
                                        ref_nf=nf_signature(ref), dev_nf=nf_signature(dev)))
                    # target itself
                    tp = pB if pB >= 0 else n_before - 1
                    st, mm, kk = classify(j.eager[tp], al[tp])
                    rec["target_res"] = st; rec["target_kind"] = kk
                    rec["per_operand"] = per
                    rec["res"] = "OK"
                    rec["operand_allzero"] = any(p["dev_allzero"] for p in per)
                    rec["operand_diverges"] = any(p["res"] == "MISMATCH" for p in per)
    except BaseException as e:
        rec["res"] = f"EXC:{type(e).__name__}:{str(e)[:80]}"
    fh.write(json.dumps(rec) + "\n")
    print(f"[opd s{a.shard}] {r['job']}:{r['out']} {r['tgt_op']} {rec.get('res')} "
          f"tgt={rec.get('target_res')} opd_allzero={rec.get('operand_allzero')} "
          f"opd_div={rec.get('operand_diverges')}", flush=True)
print("DONE", a.shard, flush=True)
