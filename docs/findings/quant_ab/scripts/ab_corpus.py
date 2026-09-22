#!/usr/bin/env python3
"""A/B two corpora generated with identical seeds, differing only in --quantize.

Reads each job's .py header:
  # delegated: backend=<b> ops=<d> non_delegated=<m> delegate_calls=<c>
  Graph: n0*=<op>(...)
Reports per-op: jobs present in float vs int8, mean delegated ops, and the
symmetric difference (ops that only lower in one mode).

usage: ab_corpus.py <float_dir> <int8_dir> [out.tsv]
"""
import re, sys, json, collections, pathlib

RE_DEL = re.compile(r"# delegated: backend=(\S+) ops=(\d+) non_delegated=(\d+) delegate_calls=(\d+)")
RE_G   = re.compile(r"Graph: (.*)")
RE_OP  = re.compile(r"n\d+\*?=([A-Za-z_][\w.]*)\(")

def scan(d):
    """{job_key: (op, deleg_ops)} — job_key is 'w3_1200', identical across the pair."""
    out = {}
    for p in pathlib.Path(d).glob("*/*.py"):
        try:
            head = p.read_text(errors="replace")[:700]
        except OSError:
            continue
        m, g = RE_DEL.search(head), RE_G.search(head)
        if not g:
            continue
        ops = RE_OP.findall(g.group(1))
        op = ops[0] if len(ops) == 1 else ("<multi:%d>" % len(ops) if ops else "<none>")
        out[p.stem] = (op, int(m.group(2)) if m else -1)
    return out

fl, q8 = sys.argv[1], sys.argv[2]
A, B = scan(fl), scan(q8)
keys = set(A) | set(B)
ops = sorted({v[0] for v in list(A.values()) + list(B.values())})

rows = []
for op in ops:
    a = [k for k in A if A[k][0] == op]
    b = [k for k in B if B[k][0] == op]
    da = [A[k][1] for k in a if A[k][1] >= 0]
    db = [B[k][1] for k in b if B[k][1] >= 0]
    both = set(a) & set(b)                      # same seed slot lowered in BOTH modes
    onlyf = set(a) - set(b)
    onlyq = set(b) - set(a)
    rows.append((op, len(a), len(b),
                 round(sum(da)/len(da), 2) if da else 0.0,
                 round(sum(db)/len(db), 2) if db else 0.0,
                 len(both), len(onlyf), len(onlyq)))

hdr = ["op", "n_float", "n_int8", "deleg_float", "deleg_int8", "both", "only_float", "only_int8"]
out = sys.argv[3] if len(sys.argv) > 3 else None
lines = ["\t".join(hdr)] + ["\t".join(str(x) for x in r) for r in rows]
if out:
    pathlib.Path(out).write_text("\n".join(lines) + "\n")

print(f"float jobs={len(A)}  int8 jobs={len(B)}  shared seed-slots={len(set(A)&set(B))}")
print(f"ops: float={len({v[0] for v in A.values()})}  int8={len({v[0] for v in B.values()})}")
qonly = [r for r in rows if r[4] != r[3]]
print(f"\nops whose mean delegated-op count CHANGED under --quantize ({len(qonly)}):")
for r in sorted(qonly, key=lambda r: -(r[4]-r[3]))[:40]:
    print(f"  {r[0]:34s} n={r[1]:5d}/{r[2]:<5d} deleg {r[3]:6.2f} -> {r[4]:6.2f}")
lost = [r for r in rows if r[1] > 0 and r[2] == 0]
gain = [r for r in rows if r[2] > 0 and r[1] == 0]
print(f"\nops present in float but ABSENT in int8 ({len(lost)}): " + ", ".join(r[0] for r in lost[:40]))
print(f"ops present in int8 but ABSENT in float ({len(gain)}): " + ", ".join(r[0] for r in gain[:40]))
