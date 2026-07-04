#!/usr/bin/env python3
"""analysis_single.md step 4 — name the mechanism per operator from device vs eager tensors."""
import json, re, math, collections

def op_of(oc):
    m=re.search(r'=([A-Za-z0-9_.]+)\(',oc); return m.group(1) if m else oc
def finite(x):
    return isinstance(x,(int,float)) and not (isinstance(x,float) and (math.isnan(x) or math.isinf(x)))
def nonfinite(x):
    return isinstance(x,float) and (math.isnan(x) or math.isinf(x))

def classify(eager, et, dtype):
    if not eager or not et or len(eager)!=len(et):
        return "SHAPE/LEN", ""
    # non-finite: et nonfinite where eager finite
    if any(nonfinite(b) and finite(a) for a,b in zip(eager,et)):
        return "NONFINITE", f"eager finite -> device {[b for b in et if nonfinite(b)][:3]}"
    # zeroed
    if all(b==0 for b in et) and any(a!=0 for a in eager):
        return "ZEROED", "device all-zero"
    # consistent scale
    ratios=[b/a for a,b in zip(eager,et) if finite(a) and finite(b) and a!=0]
    if ratios and max(ratios)-min(ratios) < 1e-3*max(1,abs(ratios[0])) and abs(ratios[0]-1)>0.05:
        return f"SCALE:{ratios[0]:.4g}", ""
    # otherwise wrong-value; magnitude of worst
    d=max((abs(b-a) for a,b in zip(eager,et) if finite(a) and finite(b)), default=0)
    return "WRONG-VALUE", f"|Δ|max={d:.4g}"

per=collections.defaultdict(list)
for line in open("findings/samsung_e9965/_work/confirmed_tensors.jsonl"):
    line=line.strip()
    if not line.startswith("{"): continue
    r=json.loads(line)
    if r.get("status")!="MISMATCH": continue
    op=op_of(r.get("op_chain",""))
    outs=r.get("outputs",[])
    if not outs: continue
    o=outs[0]
    mech,ev=classify(o.get("eager"),o.get("et"),o.get("dtype"))
    per[op].append((mech,o.get("dtype"),o.get("shape"),o.get("eager"),o.get("et"),ev,r["job_id"]))

print(f"{'operator':28} {'dtype':8} {'mechanism':14} example (eager -> device)")
for op in sorted(per):
    # dominant mechanism
    mechs=collections.Counter(x[0].split(':')[0] for x in per[op])
    dom=mechs.most_common(1)[0][0]
    rep=next((x for x in per[op] if x[0].startswith(dom)), per[op][0])
    mech,dt,shape,eg,et,ev,jid=rep
    egs=str(eg[:3]) if eg else "[]"; ets=str(et[:3]) if et else "[]"
    print(f"{op:28} {str(dt):8} {mech:14} {jid}  {egs} -> {ets}  {ev}")
