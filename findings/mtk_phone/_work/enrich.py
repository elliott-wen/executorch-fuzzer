#!/usr/bin/env python3
"""Join the feed skip-log with each job's delegated-op count and operator name.
Emits an enriched TSV: status, job_id, op, delegated_ops, non_delegated, reason
and prints per-(status,op,delegated?) summary tables.
"""
import re, sys, collections
from pathlib import Path

CORPUS = Path("mobile/corpus_v3/mtk")
SKIPLOG = Path(sys.argv[1] if len(sys.argv) > 1 else "mobile/findings/mtk_phone/_work/skip_full.tsv")
OUT = Path("mobile/findings/mtk_phone/_work/enriched.tsv")

hdr_re = re.compile(r"ops=(\d+)\s+non_delegated=(\d+)\s+delegate_calls=(\d+)")
graph_re = re.compile(r"Graph:\s*(.*)")

def job_py(jid):
    # w100:711 -> corpus/w100/w100_711.py
    shard, num = jid.split(":")
    return CORPUS / shard / f"{shard}_{num}.py"

def op_of_graph(g):
    # 'n0*=bitwise_left_shift.Tensor_Scalar_out(L0,·,·)' -> bitwise_left_shift.Tensor_Scalar_out
    g = g.strip()
    m = re.search(r"=\s*([A-Za-z0-9_.]+)\s*\(", g)
    if m: return m.group(1)
    return g

def delegated_of(jid):
    p = job_py(jid)
    if not p.exists():
        return (None, None, None)
    try:
        txt = p.read_text(errors="replace")
    except Exception:
        return (None, None, None)
    m = hdr_re.search(txt)
    if not m:
        return (None, None, None)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

rows = []
with open(SKIPLOG) as f:
    next(f, None)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        status, jid, reason, graph = parts[0], parts[1], parts[2], parts[3]
        op = op_of_graph(graph)
        ops, non, calls = delegated_of(jid)
        rows.append((status, jid, op, ops, non, reason))

with open(OUT, "w") as o:
    o.write("status\tjob\top\tdelegated_ops\tnon_delegated\treason\n")
    for status, jid, op, ops, non, reason in rows:
        o.write(f"{status}\t{jid}\t{op}\t{ops}\t{non}\t{reason}\n")

# summaries
def bucket(ops):
    if ops is None: return "unknown"
    return "delegated" if ops >= 1 else "portable"

print(f"total non-OK rows: {len(rows)}")
print("\n=== by status x delegated-bucket ===")
c = collections.Counter((r[0], bucket(r[3])) for r in rows)
for (st, bk), n in sorted(c.items()):
    print(f"  {st:9s} {bk:9s} {n}")

for STATUS in ("CRASH", "MISMATCH", "SKIP"):
    print(f"\n=== {STATUS}: DELEGATED (ops>=1) ops by count ===")
    cc = collections.Counter(r[2] for r in rows if r[0]==STATUS and (r[3] or 0) >= 1)
    for op, n in cc.most_common(40):
        print(f"  {n:5d}  {op}")
