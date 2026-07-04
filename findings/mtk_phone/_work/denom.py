import re, collections
from pathlib import Path
CORPUS = Path("mobile/corpus_v3/mtk")
hdr = re.compile(r"ops=(\d+)")
grf = re.compile(r"Graph:\s*(.*)")
opre = re.compile(r"=\s*([A-Za-z0-9_.]+)\s*\(")
tot = collections.Counter()      # (op,bucket) -> count
optot = collections.Counter()    # op -> total
nfiles = 0
for py in CORPUS.glob("*/*.py"):
    nfiles += 1
    try:
        with open(py) as f:
            l1 = f.readline(); f.readline(); l3 = f.readline()
    except Exception:
        continue
    mo = hdr.search(l1); mg = grf.search(l3)
    if not mg: 
        continue
    m2 = opre.search(mg.group(1))
    op = m2.group(1) if m2 else mg.group(1).strip()
    ops = int(mo.group(1)) if mo else -1
    bucket = "delegated" if ops>=1 else ("portable" if ops==0 else "unk")
    tot[(op,bucket)] += 1
    optot[op] += 1
print("total .py files:", nfiles)
print("distinct ops:", len(optot))
# write full denom table
with open("mobile/findings/mtk_phone/_work/denom.tsv","w") as o:
    o.write("op\ttotal\tdelegated\tportable\n")
    for op,n in optot.most_common():
        o.write(f"{op}\t{n}\t{tot[(op,'delegated')]}\t{tot[(op,'portable')]}\n")
# how many ops ever delegate at all
deleg_ops = sum(1 for op in optot if tot[(op,'delegated')]>0)
print("ops with >=1 delegated sample:", deleg_ops)
print("ops purely portable:", len(optot)-deleg_ops)
