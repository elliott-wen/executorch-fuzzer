#!/usr/bin/env python3
"""nf_adjudicate.py — the final, population-wide adjudication of all 194 NONFINITE rows."""
import json, glob, os, sys
from collections import Counter
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)


def L(pat):
    o = []
    for f in glob.glob(os.path.join(D, pat)):
        for l in open(f):
            l = l.strip()
            if l.startswith("{"):
                try:
                    o.append(json.loads(l))
                except Exception:
                    pass
    return o


K = lambda r: (r["job"], r["out"])
va = {K(r): r for r in L("va_s*.jsonl")}
anc = {K(r): r for r in L("anc_s*.jsonl")}
fin = {K(x): x for x in json.load(open(os.path.join(D, "final_rows.json")))}

CAT = {}
for k, x in fin.items():
    v = va.get(k, {})
    az = v.get("dev_allzero")
    rn = v.get("ref_nf") or {}
    ref_nonfin = (rn.get("nan", 0) + rn.get("pinf", 0) + rn.get("ninf", 0)) > 0
    a = anc.get(k, {})
    if x["verdict"] == "1_NO_VGF_DELEGATE":
        c = "A_no_vgf_delegate"
    elif x["verdict"] == "2_ALSO_DIVERGES_ON_PORTABLE":
        c = "B_diverges_without_any_delegate"
    elif x["verdict"] == "3_NO_CPU_CONTROL":
        c = "C_no_cpu_control_unfalsifiable"
    elif x["verdict"] == "4_SHARED_WITH_XNNPACK":
        c = "D_shared_with_another_delegate"
    elif az is True:
        c = "E_device_output_ALL_ZERO_is_ZEROED"
    elif x["v"] == "COMPOSITIONAL" and a.get("ancestor_diverges") is True:
        c = "F_live_ancestor_already_wrong"
    elif v.get("res") != "MISMATCH":
        c = "G_did_not_reproduce"
    else:
        c = "H_SURVIVES"
    CAT[k] = dict(cat=c, allzero=az, ref_nonfin=ref_nonfin, dtype=v.get("dtype"),
                  kind=v.get("kind"), mech_replay=v.get("mech"), res=v.get("res"),
                  sat=v.get("dev_sat65504"), ref_nf=rn, dev_nf=v.get("dev_nf"),
                  anc_div=a.get("n_anc_div"), n_anc=a.get("n_anc"))

n = len(fin)
print(f"=== FINAL ADJUDICATION, all {n} NONFINITE rows (every row measured on every axis) ===")
c = Counter(CAT[k]["cat"] for k in fin)
for kk in sorted(c):
    print(f"  {kk:38s} {c[kk]:4d}  ({100*c[kk]/n:.1f}%)")
print()
print("by original verdict:", Counter((CAT[k]["cat"], fin[k]["v"][:4]) for k in fin))
print()
print("VGF B-side reproduced (1 rep, this audit):", Counter(CAT[k]["res"] for k in fin))
print("mech() replay label:", Counter(CAT[k]["mech_replay"] for k in fin))
print("device hits fp16 limit +/-65504:", Counter(CAT[k]["sat"] for k in fin))
print("device output ALL-ZERO:", Counter(CAT[k]["allzero"] for k in fin))
print("reference itself non-finite:", Counter(CAT[k]["ref_nonfin"] for k in fin))
print("B-side output dtype:", Counter(CAT[k]["dtype"] for k in fin))
print()
surv = [k for k in fin if CAT[k]["cat"] == "H_SURVIVES"]
print(f"--- the {len(surv)} survivors ---")
print("  verdict:", Counter(fin[k]["v"] for k in surv))
print("  kind   :", Counter(CAT[k]["kind"] for k in surv))
print("  ops    :", Counter(fin[k]["tgt"] for k in surv).most_common())
print("  xnnpack delegated >0 ops (real 2nd-delegate control):",
      sum(1 for k in surv if (fin[k]["xnn_d"] or 0) > 0))
print()
print("| job | out | verdict | target | vgf deleg | ref non-finite | dev all-zero | portable B | xnnpack B (d) | category |")
print("|" + "---|" * 10)
for k in sorted(fin):
    x, m = fin[k], CAT[k]
    print(f"| {k[0]} | {k[1]} | {x['v'][:4]} | `{x['tgt']}` | {x['vgf_d']} | "
          f"{'yes' if m['ref_nonfin'] else 'no'} | {'YES' if m['allzero'] else 'no'} | "
          f"{x['port_B']} | {x['xnn_B']} ({x['xnn_d']}) | {m['cat']} |")
json.dump({f"{k[0]}|{k[1]}": dict(**CAT[k], **{f"f_{a}": b for a, b in fin[k].items()})
           for k in fin}, open(os.path.join(D, "adjudicated.json"), "w"), indent=0)
