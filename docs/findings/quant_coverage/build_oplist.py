#!/usr/bin/env python3
"""Pick ONE representative single-op graph per operator from corpus_v5/portable.
Writes oplist.tsv: op <TAB> path"""
import re, sys, pathlib, collections
CO = pathlib.Path("/data/jwen929/mobile/corpus_v5/portable")
pat = re.compile(r'^Graph:\s*n0\*?=([A-Za-z0-9_.]+)\(', re.M)
best = {}
for p in CO.rglob("w*/*.py"):
    try: head = p.read_text(errors="ignore")[:400]
    except Exception: continue
    m = pat.search(head)
    if not m: continue
    op = m.group(1)
    if op not in best:
        best[op] = p
print(f"# distinct ops: {len(best)}", file=sys.stderr)
with open("oplist.tsv", "w") as f:
    for op, p in sorted(best.items()):
        f.write(f"{op}\t{p}\n")
