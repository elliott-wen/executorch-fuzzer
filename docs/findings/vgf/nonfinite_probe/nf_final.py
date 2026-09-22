#!/usr/bin/env python3
"""nf_final.py — population-level verdict for the whole 194-row NONFINITE bucket."""
import json, glob, os
from collections import Counter
D = os.path.dirname(os.path.abspath(__file__))


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


port = {(r["job"], r["out"]): r for r in L("fp_s*.jsonl")}
xnn = {(r["job"], r["out"]): r for r in L("fx_s*.jsonl")}
vd = {(r["job"], r["out"]): r for r in L("vd_s*.jsonl")}
import sys
sys.path.insert(0, D)
from nf_common import load_rows
rows = [r for r in load_rows() if r["mechanism"] == "NONFINITE"]
rows.sort(key=lambda r: (r["job"], r["out"]))
print("bucket size:", len(rows), " portable measured:", len(port),
      " xnnpack measured:", len(xnn), " vgf-deleg measured:", len(vd))


def res(d, side):
    if not d:
        return "NOT_MEASURED"
    r = (d.get(side) or {}).get("res")
    if r is None:
        return "NOT_MEASURED"
    if r.startswith("EXC") or r.startswith("BUILD") or r.startswith("CRASH") or r == "TIMEOUT":
        return "CANT_RUN"
    return r


out = []
for r in rows:
    k = (r["job"], r["out"])
    v = vd.get(k, {})
    pv = res(port.get(k), "B")
    pa = res(port.get(k), "A")
    xv = res(xnn.get(k), "B")
    d = v.get("d")
    if d == 0:
        verdict = "1_NO_VGF_DELEGATE"
    elif pv == "MISMATCH":
        verdict = "2_ALSO_DIVERGES_ON_PORTABLE"
    elif pv == "CANT_RUN" and xv == "CANT_RUN":
        verdict = "3_NO_CPU_CONTROL"
    elif pv == "OK" and xv == "MISMATCH":
        verdict = "4_SHARED_WITH_XNNPACK"
    elif pv == "OK":
        verdict = "5_VGF_ONLY_CANDIDATE"
    else:
        verdict = "6_UNRESOLVED"
    out.append(dict(job=r["job"], out=r["out"], v=r["verdict"], tgt=r["tgt_op"],
                    vgf_d=d, port_A=pa, port_B=pv, xnn_B=xv,
                    xnn_d=(xnn.get(k, {}).get("B") or {}).get("d"), verdict=verdict))

print()
print("=== POPULATION VERDICT (all 194 NONFINITE rows) ===")
for kk, vv in sorted(Counter(x["verdict"] for x in out).items()):
    print(f"  {kk:32s} {vv:4d}  ({100*vv/len(out):.1f}%)")
print()
print("portable B-side:", Counter(x["port_B"] for x in out))
print("portable A-side:", Counter(x["port_A"] for x in out))
print("xnnpack  B-side:", Counter(x["xnn_B"] for x in out))
print("vgf deleg_ops==0:", sum(1 for x in out if x["vgf_d"] == 0),
      " ==None(unmeasured):", sum(1 for x in out if x["vgf_d"] is None))
print()
print("--- of the 'ALSO DIVERGES ON PORTABLE' rows, is the portable A side clean? ---")
sub = [x for x in out if x["verdict"] == "2_ALSO_DIVERGES_ON_PORTABLE"]
print("   n =", len(sub), Counter(x["port_A"] for x in sub))
print()
print("--- top target ops per verdict ---")
for vv in sorted(set(x["verdict"] for x in out)):
    ops = Counter(x["tgt"] for x in out if x["verdict"] == vv)
    print(f"  {vv}: {ops.most_common(8)}")
print()
print("--- VGF_ONLY_CANDIDATE rows with a REAL independent delegate control (xnnpack deleg>0 and OK) ---")
vo = [x for x in out if x["verdict"] == "5_VGF_ONLY_CANDIDATE"]
strong = [x for x in vo if (x["xnn_d"] or 0) > 0 and x["xnn_B"] == "OK"]
print(f"   VGF_ONLY total {len(vo)}; with xnnpack delegating >0 ops and OK: {len(strong)}")
print("   (the remaining %d have xnnpack deleg_ops==0, i.e. xnnpack == portable, no second delegate)"
      % (len(vo) - len(strong)))
json.dump(out, open(os.path.join(D, "final_rows.json"), "w"), indent=0)
print()
print("| job | out | vd | target | vgf d | portable A | portable B | xnnpack B (d) | verdict |")
print("|" + "---|" * 9)
for x in out:
    print(f"| {x['job']} | {x['out']} | {x['v'][:4]} | `{x['tgt']}` | {x['vgf_d']} | {x['port_A']} | "
          f"{x['port_B']} | {x['xnn_B']} ({x['xnn_d']}) | {x['verdict']} |")
