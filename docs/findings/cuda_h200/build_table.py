#!/usr/bin/env python3
"""build_table.py — join opmap.tsv + skiplog.tsv into the per-operator table + reason classes.

Single-op corpus, whole-graph AOTI: every job is delegated (verified), so every non-OK is the
CUDA delegated kernel — no portable-fallback confound to filter (step 3a is a no-op here).
Emits: per_op_table.tsv (op × {OK,MISMATCH,CRASH,SKIP,TIMEOUT}), reason_classes.tsv (signature →
count, example job_ids), and prints the summary.
"""
import re, sys
from collections import defaultdict, Counter
from pathlib import Path

D = Path("/data/jwen929/mobile/findings/cuda_h200")
opmap = {}                        # job_id -> op
total = Counter()                 # op -> total samples in corpus
for ln in (D / "opmap.tsv").read_text().splitlines()[1:]:
    job, op, _d = ln.split("\t")
    opmap[job] = op
    total[op] += 1

def classify(status, reason):
    """Map a raw reason string to a compact signature (bug-class key)."""
    r = reason
    if status == "MISMATCH":
        if "non-finite" in r: return "MISMATCH: non-finite (nan/inf positions differ)"
        if r.startswith("out[") and "shape" in r: return "MISMATCH: shape/rank divergence"
        if "max|delta|" in r: return "MISMATCH: wrong-value (max|delta|)"
        return "MISMATCH: other"
    if status == "SKIP":
        if not r.startswith("client "):
            return "SKIP(feeder): non-comparable output"      # drop — not a backend bug
        if "storage_offset must be 0" in r: return "SKIP: storage_offset!=0 (sliced/viewed input)"
        if "must be contiguous" in r: return "SKIP: non-contiguous input"
        if "dtype mismatch" in r:
            m = re.search(r"SlimTensor dtype (\d+) != ETensor dtype (\d+)", r)
            return f"SKIP: dtype mismatch ({m.group(1)}!={m.group(2)})" if m else "SKIP: dtype mismatch"
        if "undefined symbol" in r or "AOTInductorModelContainerCreateWithDevice" in r:
            return "SKIP(load): undefined symbol AOTInductorModelContainerCreateWithDevice"
        if "index out of bounds" in r: return "SKIP: device-assert index-OOB"
        if "Init failed for backend CudaBackend" in r: return "SKIP(load): CudaBackend init failed"
        # keep first kernel clause after the first '|'
        seg = r.split("|", 1)[1].strip() if "|" in r else r
        seg = re.sub(r"0x[0-9a-f]+|\d+", "N", seg)[:80]
        return f"SKIP: {seg}"
    if status == "CRASH":
        if "executor died" in r or "native abort" in r: return "CRASH: native abort (no catchable msg)"
        return "CRASH: " + re.sub(r"0x[0-9a-f]+|\d+", "N", r)[:60]
    if status == "TIMEOUT":
        return "TIMEOUT"
    return f"{status}: {r[:60]}"

per_op = defaultdict(lambda: Counter())      # op -> Counter(status)
cls_count = Counter()                        # signature -> count
cls_ex = defaultdict(list)                   # signature -> [job_ids]
cls_op = defaultdict(Counter)                # signature -> op -> count
nonok = 0
rows = (D / "skiplog.tsv").read_text().splitlines()[1:]
for ln in rows:
    parts = ln.split("\t")
    if len(parts) < 4: continue
    status, job, reason, graph = parts[0], parts[1], parts[2], parts[3]
    op = opmap.get(job, "?")
    per_op[op][status] += 1
    nonok += 1
    sig = classify(status, reason)
    cls_count[sig] += 1
    cls_op[sig][op] += 1
    if len(cls_ex[sig]) < 8:
        cls_ex[sig].append(job)

# per-op table
with open(D / "per_op_table.tsv", "w") as f:
    f.write("op\tsamples\tOK\tMISMATCH\tCRASH\tSKIP\tTIMEOUT\n")
    for op in sorted(total, key=lambda o: -(sum(per_op[o].values()))):
        c = per_op[op]
        nok = sum(c.values())
        ok = total[op] - nok
        f.write(f"{op}\t{total[op]}\t{ok}\t{c['MISMATCH']}\t{c['CRASH']}\t{c['SKIP']}\t{c['TIMEOUT']}\n")

# reason classes
with open(D / "reason_classes.tsv", "w") as f:
    f.write("count\tsignature\tn_ops\texample_job_ids\n")
    for sig, n in cls_count.most_common():
        f.write(f"{n}\t{sig}\t{len(cls_op[sig])}\t{','.join(cls_ex[sig])}\n")

print(f"non-OK rows: {nonok}  | distinct reason-classes: {len(cls_count)}")
print(f"ops with ≥1 non-OK: {len(per_op)}\n")
print("=== reason classes (count | n_ops | signature) ===")
for sig, n in cls_count.most_common():
    print(f"  {n:5d}  [{len(cls_op[sig]):3d} ops]  {sig}")
print("\n=== ops with the most non-OK (top 25) ===")
for op in sorted(per_op, key=lambda o: -sum(per_op[o].values()))[:25]:
    c = per_op[op]
    print(f"  {op:34s} MM={c['MISMATCH']:3d} CR={c['CRASH']:3d} SK={c['SKIP']:3d} TO={c['TIMEOUT']:3d}  /{total[op]}")
