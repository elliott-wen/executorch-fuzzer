import collections
E="mobile/findings/mtk_phone/_work/enriched.tsv"
D="mobile/findings/mtk_phone/_work/denom.tsv"
# non-OK delegated counts per op per status
nk=collections.defaultdict(lambda:collections.Counter())
with open(E) as f:
    next(f)
    for line in f:
        p=line.rstrip("\n").split("\t")
        if len(p)<6: continue
        st,jid,op,ops=p[0],p[1],p[2],p[3]
        try: opsn=int(ops)
        except: opsn=-1
        bucket="delegated" if opsn>=1 else "portable"
        nk[(op,bucket)][st]+=1
deleg_tot={}
with open(D) as f:
    next(f)
    for line in f:
        op,tot,dg,pt=line.rstrip("\n").split("\t")
        deleg_tot[op]=int(dg)
rows=[]
for op,dg in deleg_tot.items():
    if dg==0: continue
    c=nk[(op,"delegated")]
    mism,crash,skip=c["MISMATCH"],c["CRASH"],c["SKIP"]
    ok=dg-mism-crash-skip
    rows.append((op,dg,ok,mism,crash,skip))
# sort by (crash+mism+skip) desc
rows.sort(key=lambda r:-(r[3]+r[4]+r[5]))
print(f"{'operator':32s} {'k':>5} {'OK':>5} {'MISM':>5} {'CRASH':>5} {'SKIP':>5}")
tot=[0,0,0,0,0]
for op,dg,ok,mism,crash,skip in rows:
    print(f"{op:32s} {dg:5d} {ok:5d} {mism:5d} {crash:5d} {skip:5d}")
    for i,v in enumerate((dg,ok,mism,crash,skip)): tot[i]+=v
print(f"{'TOTAL(delegated ops)':32s} {tot[0]:5d} {tot[1]:5d} {tot[2]:5d} {tot[3]:5d} {tot[4]:5d}")
# save
with open("mobile/findings/mtk_phone/_work/optable_delegated.tsv","w") as o:
    o.write("operator\tk_delegated\tOK\tMISMATCH\tCRASH\tSKIP\n")
    for r in rows: o.write("\t".join(map(str,r))+"\n")
