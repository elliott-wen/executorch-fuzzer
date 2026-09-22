import sys, glob, collections, os
print(f"{'backend':<12}{'ops':>5}{'ANNOT':>7}{'no':>6}{'ERR':>5}{'READY':>7}{'DELEG':>7}  status")
for f in sorted(glob.glob("out_*.tsv")):
    b = f[4:-4]
    rows = [l.rstrip("\n").split("\t") for l in open(f).read().splitlines()[1:] if l.strip()]
    rows = [r for r in rows if len(r) >= 3]
    c = collections.Counter(r[1] for r in rows)
    ready = sum(1 for r in rows if len(r) > 3 and r[3] == "READY")
    deleg = sum(1 for r in rows if len(r) > 4 and r[3] == "READY" and r[4] not in ("-", "0"))
    st = "complete" if len(rows) >= 207 else f"PARTIAL {len(rows)}/209"
    print(f"{b:<12}{len(rows):>5}{c.get('YES',0):>7}{c.get('no',0):>6}{c.get('ERR',0):>5}{ready:>7}{deleg:>7}  {st}")
