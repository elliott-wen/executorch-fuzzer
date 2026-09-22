#!/usr/bin/env python3
"""nf_vgf.py — for sampled NONFINITE rows: rebuild A and B, run BOTH on the local VGF runtime,
record ref/dev non-finite signatures, leaf finiteness, N-rep determinism, and mech() replay.
Writes one TSV line per row.  --shard i --nshards N, resumable.
"""
import os, sys, json, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nf_common import *

NREP = int(os.environ.get("NREP", "3"))


def one(row):
    job_id, out_idx, verdict, needed = row["job"], row["out"], row["verdict"], row["needed"]
    srcA, srcB, target, posA, posB = srcs_for(job_id, out_idx, verdict, needed)
    rec = dict(job=job_id, out=out_idx, verdict=verdict, target=target,
               tgt_op=row["tgt_op"], trigger=row["trigger_ops"], n_div_rec=row["n_div"])
    # leaf finiteness
    lt = leaf_tensors(job_id)
    bad = {n: nf_signature(t) for n, t in lt.items()
           if t.dtype.is_floating_point and not bool(torch.isfinite(t.float()).all())}
    rec["leaf_nonfinite"] = ("YES:" + ",".join(sorted(bad))) if bad else "no"
    # --- A side ---
    jA = build_job(srcA, "vgf", False)
    if jA.status != "READY":
        rec["A"] = f"BUILD:{jA.status}"
    else:
        pA = posA if posA >= 0 else len(jA.eager) - 1
        rec["A_ref_nf"] = json.dumps(nf_signature(jA.eager[pA]))
        raws, err = vgf_run(jA, job_id.replace(":", "_") + "A")
        if raws is None:
            rec["A"] = f"RUN:{err[:60]}"
        else:
            al = align(raws, jA)
            if al is None:
                rec["A"] = "ALIGN_FAIL"
            else:
                st, m, k = classify(jA.eager[pA], al[pA])
                rec["A"] = st; rec["A_mech"] = m; rec["A_kind"] = k
                rec["A_dev_nf"] = json.dumps(nf_signature(al[pA]))
    # --- B side, N reps ---
    jB = build_job(srcB, "vgf", False)
    if jB.status != "READY":
        rec["B"] = f"BUILD:{jB.status}"
        return rec
    pB = posB if posB >= 0 else len(jB.eager) - 1
    ref = jB.eager[pB]
    rec["B_ref_nf"] = json.dumps(nf_signature(ref))
    rec["B_ref_dtype"] = str(ref.dtype).replace("torch.", "")
    vs, ms, ks, devs = [], [], [], []
    for _ in range(NREP):
        raws, err = vgf_run(jB, job_id.replace(":", "_") + "B")
        if raws is None:
            vs.append(f"RUN:{err[:40]}"); continue
        al = align(raws, jB)
        if al is None:
            vs.append("ALIGN_FAIL"); continue
        st, m, k = classify(ref, al[pB])
        vs.append(st); ms.append(m); ks.append(k); devs.append(nf_signature(al[pB]))
    rec["B"] = ";".join(vs)
    rec["B_mech"] = ";".join(sorted(set(ms)))
    rec["B_kind"] = ";".join(sorted(set(ks)))
    rec["B_ndiv"] = sum(1 for v in vs if v == "MISMATCH")
    if devs:
        rec["B_dev_nf"] = json.dumps(devs[0])
    rec["B_deleg"] = json.dumps(getattr(jB, "delegated", None) or {})
    # keep the reconstructed B source for cross-backend reuse
    rec["_srcB"] = srcB
    rec["_pB"] = pB
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--mech", default="NONFINITE")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [r for r in load_rows() if r["mechanism"] == a.mech]
    if a.mech == "BLANKQ":
        rows = [r for r in load_rows() if r["mechanism"] in ("", "?")]
    rows.sort(key=lambda r: (r["job"], r["out"]))
    if a.limit:
        # deterministic stride sample across the whole bucket
        step = max(1, len(rows) // a.limit)
        rows = rows[::step][:a.limit]
    mine = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                d = json.loads(l); done.add((d["job"], d["out"]))
            except Exception:
                pass
    fh = open(a.out, "a", buffering=1)
    for r in mine:
        if (r["job"], r["out"]) in done:
            continue
        try:
            rec = one(r)
        except Exception as e:
            rec = dict(job=r["job"], out=r["out"], verdict=r["verdict"], tgt_op=r["tgt_op"],
                       ERR=f"{type(e).__name__}:{str(e)[:120]}")
        fh.write(json.dumps(rec) + "\n")
        print(f"[s{a.shard}] {r['job']}:{r['out']} {rec.get('tgt_op')} A={rec.get('A')} "
              f"B={rec.get('B')} Bmech={rec.get('B_mech')} kind={rec.get('B_kind')} "
              f"leaf={rec.get('leaf_nonfinite')} {rec.get('ERR','')}", flush=True)
    print(f"SHARD_DONE {a.shard}", flush=True)


if __name__ == "__main__":
    main()
