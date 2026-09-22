import json,glob,collections,sys
recs=[json.loads(l) for f in glob.glob(sys.argv[1]) for l in open(f)]
def analyze(d):
    """d = one side dict from diag. returns dict of findings."""
    instrs=d["instrs"]; vals={v["val"]:v for v in d["vals"]}
    inset={i["val"] for i in d["plan_in"] if i}
    outs=[o["val"] for o in d["plan_out"] if o]
    p=d["pos"]; tv=outs[p]
    # def/last-use
    first={}; last={}
    for k,i in enumerate(instrs):
        for a in i["args"]:
            if a in vals:
                first.setdefault(a,k); last[a]=k
    N=len(instrs)
    def live(v):
        f = -1 if v in inset else first.get(v,0)
        l = N if v in set(outs) else last.get(v,f)
        return (f,l)
    # delegate in/out split
    seen=set(inset); dele=[]
    for k,i in enumerate(instrs):
        if i["k"]=="DelegateCall":
            ins=[a for a in i["args"] if a in seen]; outp=[a for a in i["args"] if a not in seen]
            dele.append({"idx":k,"in":ins,"out":outp})
        for a in i["args"]: seen.add(a)
    # producer of target
    prod=None
    for k,i in enumerate(instrs):
        if tv in i["args"]: prod=i["k"]; break
    tr=vals.get(tv)
    if tr is None: return {"err":"no_alloc"}
    lo,hi=tr["off"],tr["off"]+tr["bytes"]; tl=live(tv)
    ov_live_in=[]; ov_dead_in=[]; ov_live_int=[]; ov_dead_int=[]
    for v,w in vals.items():
        if v==tv: continue
        if w["mem_id"]!=tr["mem_id"]: continue
        if not (w["off"]<hi and lo<w["off"]+w["bytes"]): continue
        wl=live(v)
        simultaneous = (wl[0]<=tl[1] and tl[0]<=wl[1])
        isin = v in inset
        (ov_live_in if (isin and simultaneous) else ov_dead_in if isin else ov_live_int if simultaneous else ov_dead_int).append(v)
    # is target downstream of a multi-output delegate?
    multi=[dd for dd in dele if len(dd["out"])>=2]
    return {"prod":prod,"n_deleg":len(dele),"multi_out_deleg":len(multi),
            "max_deleg_outs":max([len(dd["out"]) for dd in dele],default=0),
            "ov_live_in":ov_live_in,"ov_dead_in":ov_dead_in,"ov_live_int":ov_live_int,"ov_dead_int":ov_dead_int}
tal=collections.Counter(); rows=[]
for r in recs:
    if "error" in r or "B" not in r or "instrs" not in r.get("B",{}): tal["SKIP"]+=1; continue
    a=analyze(r["B"]); aa=analyze(r["A"]) if "instrs" in r.get("A",{}) else {}
    if "err" in a: tal["NO_ALLOC"]+=1; continue
    if a["ov_live_in"]: cat="OUT_OVERLAPS_LIVE_INPUT"
    elif a["ov_live_int"]: cat="OUT_OVERLAPS_LIVE_INTERMEDIATE"
    elif a["ov_dead_in"] or a["ov_dead_int"]: cat="OWN_BUFFER(legal reuse of dead buffer)"
    else: cat="OWN_BUFFER(no overlap at all)"
    tal[cat]+=1
    tal["prod="+str(a["prod"])]+=1
    tal["multiOutDeleg>=1:"+str(a["multi_out_deleg"]>=1)]+=1
    rows.append((r["job"],r["out"],r["verdict"][:4],cat,a["prod"],a["n_deleg"],a["multi_out_deleg"],a["max_deleg_outs"],
                 aa.get("multi_out_deleg"),aa.get("max_deleg_outs")))
print("n analyzed:",len(rows))
for k,v in tal.most_common(): print(f"  {v:4d}  {k}")
print("\n--- A-side comparison: multi-output delegate present? ---")
c=collections.Counter((r[6]>=1, r[8]>=1 if r[8] is not None else None) for r in rows)
print("  (B_multi, A_multi) ->",dict(c))
print("\n--- per-case (first 60) ---")
for r in rows[:60]: print("  ",r)
json.dump(rows,open("plananal.json","w"))
