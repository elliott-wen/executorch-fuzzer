import re,sys,collections,os
sys.path.insert(0,"/data/jwen929")
from mobile.net import corpus as C
CORPUS="/data/jwen929/mobile/corpus_v4/coreml"
D="/data/jwen929/mobile/findings/coreml_mac2"
rows=[l.split("\t") for l in open(f"{D}/graphopt_mechanisms.tsv").read().splitlines()[1:] if l.strip()]

cache={}
def graph(jid):
    if jid in cache: return cache[jid]
    p=C.job_file(CORPUS,jid,"py")
    txt=open(p).read()
    # returned tuple
    m=re.search(r"return \(([^)]*)\)",txt)
    ret=[s.strip() for s in m.group(1).split(",") if s.strip()]
    ops={}
    for mm in re.finditer(r"^\s+(n\d+) = _first\(torch\.ops\.aten\.([A-Za-z0-9_.]+)\(",txt,re.M):
        ops[mm.group(1)]=mm.group(2)
    cache[jid]=(ret,ops,txt)
    return cache[jid]

tgt=collections.Counter(); trg=collections.Counter()
scale_var=0; scale_rows=[]
zero_bool=0; zero_const=0; zero_n=0
scale_single=0; scale_const=0
missing=0
per=collections.defaultdict(list)
for r in rows:
    jid,oi,verdict,req,mech,samp=(r+[""]*6)[:6]
    try: ret,ops,txt=graph(jid)
    except Exception as e:
        missing+=1; continue
    oi=int(oi)
    tname=ret[oi] if oi<len(ret) else None
    top=ops.get(tname,"LEAF/"+str(tname))
    reqops=[ops.get(x,x) for x in req.split(",") if x]
    if mech=="ZEROED":
        zero_n+=1; tgt[top]+=1
        for o in reqops: trg[o]+=1
        # eager sample values
        m=re.search(r"eager=\[([^\]]*)\]",samp)
        vals=[]
        if m:
            for v in m.group(1).split(","):
                v=v.strip()
                try: vals.append(float(v))
                except: pass
        if vals and all(v in (0.0,1.0) for v in vals): zero_bool+=1
        if vals and len(set(vals))==1: zero_const+=1
        per["ZEROED"].append((jid,oi,top,",".join(reqops),samp))
    if mech.startswith("SCALE"):
        has=("var.correction" in txt) or ("var_mean.correction" in txt) or ("std.correction" in txt)
        if has: scale_var+=1
        m=re.search(r"eager=\[([^\]]*)\]",samp)
        vals=[]
        if m:
            for v in m.group(1).split(","):
                v=v.strip()
                try: vals.append(float(v))
                except: pass
        if len(vals)==1: scale_single+=1
        if len(vals)>1 and len(set(vals))==1: scale_const+=1
        scale_rows.append((jid,oi,mech,top,",".join(reqops),"VAR" if has else "-",samp))

print("rows:",len(rows),"unresolvable job files:",missing)
print("\n=== ZEROED (%d) target-op distribution ==="%zero_n)
for o,n in tgt.most_common(40): print(f"  {n:4d}  {o}")
print("\n=== ZEROED trigger-op (required) distribution top 25 ===")
for o,n in trg.most_common(25): print(f"  {n:4d}  {o}")
print("\nZEROED with all-0/1 eager sample (bool-suspect):",zero_bool)
print("ZEROED with constant eager sample:",zero_const)
print("\n=== SCALE (%d) ==="%len(scale_rows))
print("  graphs containing var/std.correction:",scale_var)
print("  single-element eager sample (ratio test trivial):",scale_single)
print("  multi-element but constant eager sample:",scale_const)
for r in sorted(scale_rows,key=lambda x:x[2]): print("   ",r)
