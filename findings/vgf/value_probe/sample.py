#!/usr/bin/env python3
"""Build the audit sample: 40 VALUE rows (stratified) + ALL SCALE:*/ALIAS:* rows.
Joins graphopt_all.tsv (mechanism) with bisect_results.tsv (needed_siblings/needed_live, deleg)."""
import random, json, sys
ROOT = "/data/jwen929/mobile"
G = f"{ROOT}/findings/vgf/graphopt_all.tsv"
B = f"{ROOT}/findings/vgf/bisect_results.tsv"

bis = {}
with open(B) as f:
    hdr = f.readline().rstrip("\n").split("\t"); ix = {c: i for i, c in enumerate(hdr)}
    for line in f:
        r = line.rstrip("\n").split("\t")
        if len(r) <= ix["detail"]:
            continue
        bis[(r[0], r[1])] = r
BI = ix

rows = []
HDR = ["job","out","verdict","target_op","trigger_ops","ab","n_div","n_rep","mechanism"]
with open(G) as f:
    hdr = HDR
    for line in f:
        r = line.rstrip("\n").split("\t")
        if len(r) < 9:
            continue
        rows.append(dict(zip(hdr, r)))

def enrich(r):
    b = bis.get((r["job"], r["out"]))
    if b is None:
        return None
    v = r["verdict"]
    need = b[BI["needed_siblings"]] if v == "SIBLING_DEPENDENT" else b[BI["needed_live"]]
    return {"job": r["job"], "out": int(r["out"]), "verdict": v, "mech": r["mechanism"],
            "ab": r["ab"], "n_div": r["n_div"], "tgt_op": r["target_op"],
            "trigger_ops": r["trigger_ops"],
            "needed": [x for x in need.split(",") if x],
            "deleg_full": b[BI["deleg_full"]], "deleg_tgt": b[BI["deleg_tgt"]],
            "target": b[BI["target"]]}

value = [x for x in (enrich(r) for r in rows if r["mechanism"] == "VALUE") if x]
sa = [x for x in (enrich(r) for r in rows if r["mechanism"].startswith(("SCALE", "ALIAS"))) if x]
comp = [x for x in value if x["verdict"] == "COMPOSITIONAL"]
sib = [x for x in value if x["verdict"] == "SIBLING_DEPENDENT"]
random.seed(20260823)
sel = random.sample(comp, 30) + random.sample(sib, 12)
for x in sel:
    x["bucket"] = "VALUE"
for x in sa:
    x["bucket"] = "SCALE_ALIAS"
out = sel + sa
print(f"VALUE pool comp={len(comp)} sib={len(sib)}; sampled={len(sel)}; scale/alias={len(sa)}; total={len(out)}", file=sys.stderr)
json.dump(out, open(sys.argv[1], "w"), indent=0)
