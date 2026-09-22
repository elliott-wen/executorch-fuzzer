#!/usr/bin/env python3
"""nf_table.py — join the VGF A/B run, the cross-backend runs and the delegation counts
into the audit's master per-case table + the VGF-only vs shared split."""
import json, os
from collections import Counter
D = os.path.dirname(os.path.abspath(__file__))


def load(f):
    out = []
    for l in open(os.path.join(D, f)):
        l = l.strip()
        if not l.startswith("{"):
            continue
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


nf = {(r["job"], r["out"]): r for r in load("nf_all.jsonl")}
dg = {b: {(r["job"], r["out"]): r for r in load(f"dg_{b}.jsonl")}
      for b in ["portable", "xnnpack", "openvino", "vgf"]}
xb = {b: {(r["job"], r["out"]): r for r in load(f"xb_{b}.jsonl") if "job" in r}
      for b in ["portable", "xnnpack", "openvino"]}


def short(res):
    if res is None:
        return "-"
    if res.startswith("EXC:SystemExit"):
        return "SKIP"
    if res.startswith("EXC"):
        return "ERR"
    return {"MISMATCH": "DIVERGES", "OK": "OK"}.get(res, res[:8])


rows = []
for k, r in sorted(nf.items()):
    vd = dg["vgf"].get(k, {}).get("d", -1)
    xd = dg["xnnpack"].get(k, {}).get("d", -1)
    od = dg["openvino"].get(k, {}).get("d", -1)
    p = short(xb["portable"].get(k, {}).get("res"))
    x = short(xb["xnnpack"].get(k, {}).get("res"))
    o = short(xb["openvino"].get(k, {}).get("res"))
    # direction of the VGF non-finite divergence
    kind = r.get("B_kind", "")
    direction = ("dev_invents" if kind == "dev_nonfinite" else
                 "ref_saturated" if kind == "ref_nonfinite" else "both")
    # verdict
    if vd == 0:
        v = "NO_VGF_DELEGATE"
    elif p == "DIVERGES":
        v = "REF_PIPELINE (portable too)"
    elif p == "ERR":
        v = "PORTABLE_CANT_RUN (inconclusive)"
    elif x == "DIVERGES" or o == "DIVERGES":
        v = "SHARED_WITH_DELEGATE"
    else:
        v = "VGF_ONLY"
    rows.append(dict(job=k[0], out=k[1], verdict=r["verdict"][:4], tgt=r.get("tgt_op"),
                     dtype=r.get("B_ref_dtype"), vgf_d=vd, xnn_d=xd, ov_d=od,
                     A=r.get("A"), ndiv=f"{r.get('B_ndiv')}/3", rec_ndiv=r.get("n_div_rec"),
                     leaf=r.get("leaf_nonfinite"), direction=direction,
                     port=p, xnn=x, ov=o, V=v,
                     ref_nf=r.get("B_ref_nf"), dev_nf=r.get("B_dev_nf")))

hdr = ("| job | out | vd | target op | out dtype | vgf/xnn/ov deleg_ops | A alone | VGF B n_div | "
       "leaf finite? | direction | portable | xnnpack | openvino | verdict |")
print(hdr)
print("|" + "---|" * 14)
for r in rows:
    print(f"| {r['job']} | {r['out']} | {r['verdict']} | `{r['tgt']}` | {r['dtype']} | "
          f"{r['vgf_d']}/{r['xnn_d']}/{r['ov_d']} | {r['A']} | {r['ndiv']} | "
          f"{'yes' if r['leaf'] == 'no' else 'NO'} | {r['direction']} | {r['port']} | {r['xnn']} | "
          f"{r['ov']} | {r['V']} |")
print()
print("VERDICT SPLIT:", Counter(r["V"] for r in rows))
print("DIRECTION:", Counter(r["direction"] for r in rows))
print("VERDICT x DIRECTION:", Counter((r["direction"], r["V"]) for r in rows))
print("portable:", Counter(r["port"] for r in rows))
print("xnnpack :", Counter(r["xnn"] for r in rows))
print("openvino:", Counter(r["ov"] for r in rows))
print("recorded n_div:", Counter(r["rec_ndiv"] for r in rows), " reproduced n_div:", Counter(r["ndiv"] for r in rows))
print("A side:", Counter(r["A"] for r in rows))
print("leaf non-finite:", Counter(r["leaf"] for r in rows))
print("out dtype:", Counter(r["dtype"] for r in rows))
print("vgf deleg_ops==0:", [f"{r['job']}:{r['out']}" for r in rows if r["vgf_d"] == 0])
print("xnn deleg_ops==0:", sum(1 for r in rows if r["xnn_d"] == 0))
