#!/usr/bin/env python3
"""analyze.py — join the VGF single-op feed skip-log with the delegation breakdown, bucket
every non-OK outcome per operator, and emit the per-operator table + confound-filtered
worklists. Implements steps 1-3 of analysis_single.md (feed already done → skiplog.tsv).

Run:  PYTHONPATH=/data/jwen929 .venv/bin/python findings/vgf_single/analyze.py
"""
from __future__ import annotations
import re, sys, json, collections
from pathlib import Path

ROOT = Path("/data/jwen929/mobile")
CORPUS = ROOT / "corpus_v3/vgf"
OUT = ROOT / "findings/vgf_single"
SKIPLOG = OUT / "skiplog.tsv"

OP_RE = re.compile(r"n0\*?=([A-Za-z0-9_.]+?)\(")
DELEG_RE = re.compile(r"ops=(\d+)")

def op_of(graph: str) -> str:
    m = OP_RE.search(graph or "")
    return m.group(1) if m else "?"

_deleg_cache: dict[str, int | None] = {}
def deleg_ops(job_id: str) -> int | None:
    """Delegated-op count for a job_id 'w12:345' from its .py header; None if unknown."""
    if job_id in _deleg_cache:
        return _deleg_cache[job_id]
    try:
        w, idx = job_id.split(":")
        p = CORPUS / w / f"{w}_{idx}.py"
        head = p.read_text().split("\n", 1)[0]
        m = DELEG_RE.search(head)
        v = int(m.group(1)) if m else None
    except Exception:
        v = None
    _deleg_cache[job_id] = v
    return v

def classify_skip(reason: str) -> str:
    """runtime SKIP (backend rejected, coverage gap) vs feeder-side SKIP (drop, not a bug)."""
    r = reason or ""
    if r.startswith("output decode") or "buffer length" in r or "client decode" in r:
        return "skip_feeder"
    return "skip_runtime"

def main() -> int:
    if not SKIPLOG.exists():
        sys.exit(f"no skip-log at {SKIPLOG}")
    rows = []
    with open(SKIPLOG) as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            status, job, reason, graph = parts[0], parts[1], parts[2], parts[3]
            rows.append((status, job, reason, graph))

    # per-op aggregation
    ops = collections.defaultdict(lambda: collections.Counter())
    op_reasons = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    buckets = {"mismatch_delegated": [], "mismatch_portable": [],
               "crash_delegated": [], "crash_portable": [],
               "skip_runtime": [], "skip_feeder": []}

    for status, job, reason, graph in rows:
        op = op_of(graph)
        d = deleg_ops(job)
        deleg = (d is not None and d >= 1)
        if status == "MISMATCH":
            key = "mismatch_delegated" if deleg else "mismatch_portable"
            buckets[key].append((op, job, reason, d))
            ops[op][key] += 1
            op_reasons[op]["MISMATCH"][_mism_kind(reason)] += 1
        elif status == "CRASH":
            key = "crash_delegated" if deleg else "crash_portable"
            buckets[key].append((op, job, reason, d))
            ops[op][key] += 1
            op_reasons[op]["CRASH"][reason[:60]] += 1
        elif status == "SKIP":
            sk = classify_skip(reason)
            buckets[sk].append((op, job, reason, d))
            ops[op][sk] += 1
            op_reasons[op]["SKIP"][_skip_kind(reason)] += 1
        elif status == "TIMEOUT":
            ops[op]["timeout"] += 1
            buckets.setdefault("timeout", []).append((op, job, reason, d))

    # write per-op table
    with open(OUT / "per_op_table.tsv", "w") as f:
        f.write("operator\tmism_deleg\tmism_portable\tcrash_deleg\tcrash_portable\tskip_runtime\tskip_feeder\ttimeout\n")
        for op in sorted(ops, key=lambda o: -sum(ops[o].values())):
            c = ops[op]
            f.write(f"{op}\t{c['mismatch_delegated']}\t{c['mismatch_portable']}\t"
                    f"{c['crash_delegated']}\t{c['crash_portable']}\t{c['skip_runtime']}\t"
                    f"{c['skip_feeder']}\t{c['timeout']}\n")

    # dump buckets as JSON for the next stages (determinism gate)
    with open(OUT / "buckets.json", "w") as f:
        json.dump({k: v for k, v in buckets.items()}, f, indent=1)

    # summary
    tot = collections.Counter()
    for status, *_ in rows:
        tot[status] += 1
    print("=== non-OK rows:", len(rows), dict(tot))
    for k, v in buckets.items():
        print(f"  {k:20s}: {len(v)}  ops={len(set(x[0] for x in v))}")
    # the finding worklists (delegated only per 3a)
    print("\n=== MISMATCH on VGF delegate (ops>=1) — by operator ===")
    md = collections.Counter(x[0] for x in buckets["mismatch_delegated"])
    for op, n in md.most_common():
        print(f"  {n:4d}  {op}")
    print("\n=== CRASH on VGF delegate (ops>=1) — by operator ===")
    cd = collections.Counter(x[0] for x in buckets["crash_delegated"])
    for op, n in cd.most_common():
        print(f"  {n:4d}  {op}")
    return 0

def _mism_kind(reason: str) -> str:
    if "non-finite" in reason:
        return "nonfinite"
    if "shape" in reason.lower():
        return "shape"
    if "dtype" in reason.lower():
        return "dtype"
    return "value"

def _skip_kind(reason: str) -> str:
    r = reason or ""
    if "rc=2" in r:
        return "runtime-reject(rc=2)"
    if "output decode" in r or "buffer length" in r:
        return "feeder-decode"
    return r[:50]

if __name__ == "__main__":
    raise SystemExit(main())
