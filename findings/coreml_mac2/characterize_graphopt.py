#!/usr/bin/env python3
"""characterize_graphopt.py — name the MECHANISM of each graph-opt (SIBLING_DEPENDENT/COMPOSITIONAL)
CoreML finding (analysis.md step 3): replay the full graph on the Mac, read the target output's
device vs eager values, and classify ZEROED / NONFINITE / SCALE:r / ALIAS / WRONG-VALUE.

Reads bisect_results.*.tsv, replays each graph-opt (job,out) via `mobile feed <job_id>` (rich JSON),
writes graphopt_mechanisms.tsv. Needs broker + Mac worker up.
"""
import json, subprocess, sys, glob, os, re
from pathlib import Path
MOBILE="/data/jwen929/mobile"; REPO="/data/jwen929"
D=Path(f"{MOBILE}/findings/coreml_mac2")

def gather():
    seen={}
    for f in glob.glob(f"{D}/bisect_results.*.tsv"):
        for ln in open(f).read().splitlines()[1:]:
            c=ln.split("\t")
            if len(c)>=5 and c[2] in ("SIBLING_DEPENDENT","COMPOSITIONAL"):
                seen[(c[0],int(c[1]))]=(c[2],c[4])   # verdict, required
    return seen

def replay(job_ids):
    cmd=[f"{MOBILE}/.venv/bin/python","-m","mobile","feed",*sorted({j for j,_ in job_ids}),
         "--corpus",f"{MOBILE}/corpus_v4/coreml","--host","127.0.0.1","--job-port","15554",
         "--ctrl-port","15556","--timeout","90"]
    e={**os.environ,"PYTHONPATH":REPO}
    p=subprocess.run(cmd,cwd=REPO,env=e,capture_output=True,text=True,timeout=3600)
    recs={}
    for ln in p.stdout.splitlines():
        ln=ln.strip()
        if ln.startswith("{"):
            try:
                r=json.loads(ln); recs[r["job_id"]]=r
            except Exception: pass
    return recs

def classify(o):
    """o = one output dict {eager:[...], et:[...], shape,...} → mechanism string."""
    eager=o.get("eager"); et=o.get("et")
    if eager is None or et is None:
        # shape-only diff record (no values) → metadata divergence
        return f"WRONG-SHAPE {o.get('shape')} vs {o.get('et_shape')}"
    def nf(v): return isinstance(v,(int,float)) and (v!=v or abs(v)==float('inf'))
    def finite(x): return [v for v in x if isinstance(v,(int,float)) and not nf(v)]
    # reference-degenerate: the EAGER side is already non-finite → not a device dropped/plan bug (3d)
    if any(nf(v) for v in eager):
        return "REF-NONFINITE (filter)"
    if all((isinstance(v,(int,float)) and abs(v)<1e-12) for v in et) and any(abs(v)>1e-9 for v in finite(eager)):
        return "ZEROED"
    if any(nf(v) for v in et):
        return "NONFINITE"
    # SCALE: consistent ratio et/eager (incl. single repeated-value outputs)
    ratios=[e/g for g,e in zip(eager,et) if isinstance(g,(int,float)) and abs(g)>1e-6 and isinstance(e,(int,float))]
    if ratios:
        r0=ratios[0]
        if all(abs(r-r0)<0.05*max(1,abs(r0)) for r in ratios) and abs(r0-1.0)>0.08:
            return f"SCALE:{r0:.3g}"
    # near-tolerance? max rel-delta
    rels=[abs(e-g)/max(1e-9,abs(g)) for g,e in zip(eager,et) if isinstance(g,(int,float)) and isinstance(e,(int,float))]
    if rels and max(rels)<0.03:
        return "NEAR-TOL (gate N>=5)"
    return "WRONG-VALUE"

def main():
    go=gather()
    print(f"graph-opt cases to characterize: {len(go)}")
    recs=replay(list(go.keys()))
    rows=[]
    for (jid,oi),(verdict,req) in sorted(go.items()):
        r=recs.get(jid)
        if not r or "outputs" not in r:
            rows.append((jid,oi,verdict,req,"NO-REPLAY","")); continue
        outs=r["outputs"]
        o=next((x for x in outs if x.get("idx")==oi),None)
        if o is None:
            rows.append((jid,oi,verdict,req,"NO-OUTPUT","")); continue
        mech=classify(o)
        samp=f"eager={ (o.get('eager') or ['shape:'+str(o.get('shape'))])[:4] } et={ (o.get('et') or ['shape:'+str(o.get('et_shape'))])[:4] }"
        rows.append((jid,oi,verdict,req,mech,samp))
    with open(D/"graphopt_mechanisms.tsv","w") as f:
        f.write("job_id\tout\tverdict\trequired\tmechanism\tsample\n")
        for r in rows: f.write("\t".join(str(x) for x in r)+"\n")
    from collections import Counter
    c=Counter(r[4] for r in rows)
    print("=== mechanism distribution ===");
    for m,n in c.most_common(): print(f"  {n:3d}  {m}")
    print("\n=== sample rows ===")
    for r in rows[:25]: print(f"  {r[0]:12s} out[{r[1]}] {r[2]:18s} req={r[3]:10s} {r[4]:14s} {r[5]}")

if __name__=="__main__":
    main()
