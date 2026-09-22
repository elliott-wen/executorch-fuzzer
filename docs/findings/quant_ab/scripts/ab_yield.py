#!/usr/bin/env python3
"""Paired LOWERING yield: float vs int8, restricted to seed slots BOTH fleets attempted.

pregen writes _stats/<worker>.json with an `outcomes` map index -> stage (ready / lower /
export / eager / emit / gen_fail / crashed). The two fleets ran the same seed schedule, so
intersecting the attempted indices gives an exact per-graph comparison of what --quantize
does to lowering, with no bias from the fleets being stopped at different points.

usage: ab_yield.py <float_corpus> <int8_corpus>
"""
import sys, json, collections, pathlib
FC, QC = sys.argv[1], sys.argv[2]

def load(d):
    out = {}
    for p in (pathlib.Path(d) / "_stats").glob("*.json"):
        j = json.loads(p.read_text())
        w = j["producer_id"]
        for idx, stage in j.get("outcomes", {}).items():
            out[f"{w}:{idx}"] = stage
    return out

F, Q = load(FC), load(QC)
both = sorted(set(F) & set(Q))
print(f"float attempts={len(F)}  int8 attempts={len(Q)}  attempted by BOTH={len(both)}")
fr = sum(1 for k in both if F[k] == "ready"); qr = sum(1 for k in both if Q[k] == "ready")
print(f"  READY: float {fr} ({100*fr/len(both):.1f}%)   int8 {qr} ({100*qr/len(both):.1f}%)   Δ={qr-fr:+d}")
t = collections.Counter((F[k], Q[k]) for k in both)
print("\ntransition (float stage -> int8 stage), top 12:")
for (a, b), n in t.most_common(12):
    mark = "" if a == b else ("   <-- lost by quantizing" if a == "ready" else
                              ("   <-- gained by quantizing" if b == "ready" else ""))
    print(f"  {a:9s} -> {b:9s} {n:6d}{mark}")
lost = [k for k in both if F[k] == "ready" and Q[k] != "ready"]
print(f"\nlost {len(lost)} graphs to the quantized path; failing stage:",
      dict(collections.Counter(Q[k] for k in lost)))
