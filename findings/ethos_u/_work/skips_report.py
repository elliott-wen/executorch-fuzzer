#!/usr/bin/env python3
"""Aggregate SKIP and CRASH reasons per operator, split delegated vs portable.
The reason string IS the finding (analysis_single.md step 4). Normalizes volatile bits
(tmp paths, hashes) so distinct reason *signatures* group together.
"""
import re, collections

W = "/data/jwen929/mobile/findings/ethos_u/_work"
man = {}
for ln in open(f"{W}/manifest.tsv").read().splitlines()[1:]:
    f = ln.split("\t")
    if len(f) >= 3:
        man[f[0]] = (f[1], f[2])

def sig(reason):
    r = reason
    r = re.sub(r"/tmp/\S+", "/tmp/…", r)
    r = re.sub(r"ops=[0-9a-f]{6,}", "ops=…", r)
    r = re.sub(r"0x[0-9a-fp]+", "0x…", r)
    r = re.sub(r"\b\d+\b", "N", r)
    return r.strip()[:200]

skip_by = collections.defaultdict(lambda: collections.Counter())   # (bucket,op) -> sig counter
crash_by = collections.defaultdict(lambda: collections.Counter())
for ln in open(f"{W}/skip.tsv").read().splitlines()[1:]:
    f = ln.split("\t")
    if len(f) < 2:
        continue
    st, jid = f[0], f[1]
    reason = f[2] if len(f) > 2 else ""
    op, d = man.get(jid, ("?", "-2"))
    bucket = "delegated" if d not in ("0", "-2") else ("portable" if d == "0" else "unknown")
    if st == "SKIP":
        skip_by[(bucket, op)][sig(reason)] += 1
    elif st == "CRASH":
        crash_by[(bucket, op)][sig(reason)] += 1

def dump(title, tbl):
    print(f"\n{'='*80}\n{title}\n{'='*80}")
    for bucket in ("delegated", "portable", "unknown"):
        keys = sorted(k for k in tbl if k[0] == bucket)
        if not keys:
            continue
        print(f"\n--- {bucket} ---")
        for (b, op) in keys:
            for s, c in tbl[(b, op)].most_common():
                print(f"  {c:>4}  {op:<34} {s}")

dump("SKIP reasons (per op, per reason signature)", skip_by)
dump("CRASH reasons (per op, per reason signature)", crash_by)
