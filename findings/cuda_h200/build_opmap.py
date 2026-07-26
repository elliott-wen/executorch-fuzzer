#!/usr/bin/env python3
"""build_opmap.py — walk the CUDA single-op corpus and emit (job_id, op, delegated_ops) per job.

For --nodes 1 each graph is one source operator wired to leaves; the op is the first (only) real
node in the `.py` "Graph:" header line. We also pull `delegated ops=` from the header (the
delegated-vs-portable triage signal). Writes opmap.tsv and prints the op→count tally + total.
"""
import re, sys
from pathlib import Path
from collections import Counter

CORPUS = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/jwen929/mobile/corpus_v3/cuda")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/data/jwen929/mobile/findings/cuda_h200/opmap.tsv")

# "Graph: n0*=_adaptive_avg_pool2d(L0,·)"  → op = _adaptive_avg_pool2d
op_re = re.compile(r"Graph:\s*n\d+\*?=([\w.]+)\(")
deleg_re = re.compile(r"delegated:.*\bops=(\d+)")

rows = []
op_count = Counter()
deleg_count = Counter()   # op -> #jobs with delegated ops>=1
n_files = 0
for py in CORPUS.glob("*/*.py"):
    if "_crashes" in py.parts:
        continue
    n_files += 1
    op = None; dops = 0
    try:
        with open(py) as f:
            head = [next(f, "") for _ in range(6)]
    except Exception:
        continue
    for ln in head:
        m = op_re.search(ln)
        if m and op is None:
            op = m.group(1)
        d = deleg_re.search(ln)
        if d:
            dops = int(d.group(1))
    if op is None:
        continue
    # filename wN_M.py under producer dir wN  →  job_id "wN:M"
    prod = py.parent.name
    idx = py.stem.split("_", 1)[1] if "_" in py.stem else py.stem
    job_id = f"{prod}:{idx}"
    rows.append((job_id, op, dops))
    op_count[op] += 1
    if dops >= 1:
        deleg_count[op] += 1

with open(OUT, "w") as f:
    f.write("job_id\top\tdelegated_ops\n")
    for job_id, op, dops in rows:
        f.write(f"{job_id}\t{op}\t{dops}\n")

print(f"scanned {n_files} .py files, mapped {len(rows)} jobs, {len(op_count)} distinct ops")
print(f"opmap → {OUT}")
print("\ntop 25 ops by sample count (delegated = jobs the CUDA delegate absorbed):")
for op, c in op_count.most_common(25):
    print(f"  {c:5d}  deleg={deleg_count[op]:5d}  {op}")
# fully-portable ops (never delegated) — a coverage note
never = sorted(op for op in op_count if deleg_count[op] == 0)
print(f"\nops that NEVER delegated (ops=0 everywhere, portable fallback): {len(never)}")
print("  " + ", ".join(never[:40]) + (" ..." if len(never) > 40 else ""))
