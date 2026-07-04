import json,collections
# denom: delegated totals
deleg={}
with open("mobile/findings/mtk_phone/_work/denom.tsv") as f:
    next(f)
    for line in f:
        op,tot,dg,pt=line.rstrip("\n").split("\t"); deleg[op]=int(dg)
# original bulk non-OK (enriched), delegated only
mis_val=collections.Counter(); mis_nf=collections.Counter(); skip=collections.Counter(); bulk_crash=collections.Counter()
with open("mobile/findings/mtk_phone/_work/enriched.tsv") as f:
    next(f)
    for line in f:
        p=line.rstrip("\n").split("\t")
        if len(p)<6: continue
        st,jid,op,ops,non,reason=p
        if ops=="" or int(ops)<1: continue   # delegated only
        if st=="MISMATCH":
            (mis_nf if "non-finite" in reason else mis_val)[op]+=1
        elif st=="SKIP": skip[op]+=1
        elif st=="CRASH": bulk_crash[op]+=1
# serial crash verdicts (re-run of the 4517 bulk crashes)
ser_ok=collections.Counter(); ser_crash=collections.Counter(); ser_mv=collections.Counter(); ser_mnf=collections.Counter()
for l in open("mobile/findings/mtk_phone/_work/crash_rerun.jsonl"):
    l=l.strip()
    if not l.startswith("{"): continue
    d=json.loads(l)
    o=d['op_chain'].split('=')[1].split('(')[0] if '=' in d['op_chain'] else d['op_chain']
    st=d['status']
    if st=="OK": ser_ok[o]+=1
    elif st=="CRASH": ser_crash[o]+=1
    elif st=="MISMATCH":
        if "non-finite" in (d.get('detail') or ''): ser_mnf[o]+=1
        else: ser_mv[o]+=1
# combine
ops=sorted([o for o,dg in deleg.items() if dg>0])
rows=[]
for op in ops:
    dg=deleg[op]
    mv=mis_val[op]+ser_mv[op]         # value mismatches (bulk + recovered)
    mnf=mis_nf[op]+ser_mnf[op]        # nonfinite mismatches (confound)
    sk=skip[op]
    cr=ser_crash[op]                  # serial-confirmed crashes
    ok=dg-mv-mnf-sk-cr
    rows.append((op,dg,ok,mv,mnf,cr,sk))
rows.sort(key=lambda r:-(r[3]+r[5]+r[6]))   # by (value-mism + crash + skip)
hdr=f"{'operator':26s} {'k':>4} {'OK':>4} {'MvalG':>5} {'Mnf':>4} {'CRSH':>4} {'SKIP':>4}"
print(hdr); print("-"*len(hdr))
tot=[0]*6
lines=[hdr,"-"*len(hdr)]
for op,dg,ok,mv,mnf,cr,sk in rows:
    line=f"{op:26s} {dg:4d} {ok:4d} {mv:5d} {mnf:4d} {cr:4d} {sk:4d}"
    print(line); lines.append(line)
    for i,v in enumerate((dg,ok,mv,mnf,cr,sk)): tot[i]+=v
t=f"{'TOTAL':26s} {tot[0]:4d} {tot[1]:4d} {tot[2]:5d} {tot[3]:4d} {tot[4]:4d} {tot[5]:4d}"
print(t); lines.append(t)
print("\nLegend: MvalG=value-mismatch(candidate bug)  Mnf=nonfinite-mismatch(reference/leaf confound, ruled out)  CRSH=serial-confirmed crash  SKIP=runtime reject")
open("mobile/findings/mtk_phone/_work/final_optable.txt","w").write("\n".join(lines)+"\n")
