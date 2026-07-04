#!/usr/bin/env python3
"""Join manifest.tsv + skip.tsv -> per-operator non-OK table, split delegated vs portable.

Single-op corpus: every job is one operator. Step 3a of analysis_single.md: a MISMATCH/CRASH
on a DELEGATED job (delegated_ops>=1) is an ethos-u backend-kernel finding; the same on a
PORTABLE job (ops=0) is a portable-CPU/reference issue, NOT an ethos-u bug.

Usage: analyze.py            # print tables to stdout
"""
import collections
from pathlib import Path

W = Path("/data/jwen929/mobile/findings/ethos_u/_work")
MAN = W / "manifest.tsv"
SKIP = W / "skip.tsv"

# job_id -> (op, delegated_ops)
man = {}
op_total = collections.Counter()
op_deleg_total = collections.Counter()   # per-op count of delegated jobs
op_port_total = collections.Counter()
for ln in MAN.read_text().splitlines()[1:]:
    f = ln.split("\t")
    if len(f) < 3:
        continue
    jid, op, ops = f[0], f[1], f[2]
    try:
        d = int(ops)
    except ValueError:
        d = -1
    man[jid] = (op, d)
    op_total[op] += 1
    if d >= 1:
        op_deleg_total[op] += 1
    elif d == 0:
        op_port_total[op] += 1

# skip.tsv: status \t job \t reason \t graph  (non-OK only)
rows = []
if SKIP.exists():
    for ln in SKIP.read_text().splitlines()[1:]:
        f = ln.split("\t")
        if len(f) < 2:
            continue
        status, jid = f[0], f[1]
        reason = f[2] if len(f) > 2 else ""
        op, d = man.get(jid, ("?", -2))
        rows.append((status, jid, reason, op, d))

# per (op, bucket, status)
Key = collections.namedtuple("Key", "op bucket status")
cnt = collections.Counter()
for status, jid, reason, op, d in rows:
    bucket = "delegated" if d >= 1 else ("portable" if d == 0 else "unknown")
    cnt[Key(op, bucket, status)] += 1

def section(title, bucket):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    ops = sorted({k.op for k in cnt if k.bucket == bucket})
    print(f"{'operator':<34}{'MISM':>6}{'CRASH':>6}{'SKIP':>6}{'TMO':>6}{'jobs':>7}")
    tot = collections.Counter()
    for op in ops:
        m = cnt[Key(op, bucket, "MISMATCH")]
        c = cnt[Key(op, bucket, "CRASH")]
        s = cnt[Key(op, bucket, "SKIP")]
        t = cnt[Key(op, bucket, "TIMEOUT")]
        jt = op_deleg_total[op] if bucket == "delegated" else op_port_total[op]
        print(f"{op:<34}{m:>6}{c:>6}{s:>6}{t:>6}{jt:>7}")
        tot["MISMATCH"] += m; tot["CRASH"] += c; tot["SKIP"] += s; tot["TIMEOUT"] += t
    print(f"{'-'*64}")
    print(f"{'TOTAL':<34}{tot['MISMATCH']:>6}{tot['CRASH']:>6}{tot['SKIP']:>6}{tot['TIMEOUT']:>6}")

print(f"corpus: {len(man)} jobs, {len(op_total)} ops "
      f"({sum(op_deleg_total.values())} delegated, {sum(op_port_total.values())} portable)")
print(f"non-OK rows in skip.tsv so far: {len(rows)}")
by_status = collections.Counter(r[0] for r in rows)
by_bucket = collections.Counter(("delegated" if r[4] >= 1 else "portable" if r[4]==0 else "unk") for r in rows)
print(f"  by status: {dict(by_status)}")
print(f"  by bucket: {dict(by_bucket)}")

section("DELEGATED (ops>=1) — candidate ETHOS-U backend findings", "delegated")
section("PORTABLE (ops=0) — NOT ethos-u (portable-CPU/reference), ruled out of ethos-u", "portable")
