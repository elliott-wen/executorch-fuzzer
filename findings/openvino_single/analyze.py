#!/usr/bin/env python3
"""analyze.py — steps 1-3 of analysis_single.md for the OpenVINO single-op corpus.

Joins the feed skip-log (non-OK outcomes) with the corpus manifest (op + delegation per job,
the coverage denominator), buckets every non-OK outcome per operator, applies the step-3a
delegated-vs-portable split, and emits:
  - per_op_table.tsv : one row per operator, k samples + OK/MISM/CRASH/SKIP + delegated/portable
  - buckets.json     : the confound-filtered worklists for the determinism gate
  - coverage.json    : ops produced vs ops with a non-OK verdict
Prints the delegated-mismatch and delegated-crash worklists (the OpenVINO findings).

Run:  PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/analyze.py
"""
from __future__ import annotations
import sys, json, collections
from pathlib import Path

ROOT = Path("/data/jwen929/mobile")
OUT = ROOT / "findings/openvino_single"
SKIPLOG = OUT / "skiplog.tsv"
MANIFEST = OUT / "manifest.tsv"


def load_manifest():
    """job_id -> (op, deleg_ops|None); also per-op sample counts + per-op deleg/portable counts."""
    j2op = {}
    j2deleg = {}
    per_op_total = collections.Counter()
    per_op_deleg = collections.Counter()
    per_op_portable = collections.Counter()
    per_op_unknown = collections.Counter()
    with open(MANIFEST) as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            jid, op, ops = p[0], p[1], p[2]
            j2op[jid] = op
            per_op_total[op] += 1
            if ops == "":
                j2deleg[jid] = None
                per_op_unknown[op] += 1
            else:
                d = int(ops)
                j2deleg[jid] = d
                if d >= 1:
                    per_op_deleg[op] += 1
                else:
                    per_op_portable[op] += 1
    return j2op, j2deleg, per_op_total, per_op_deleg, per_op_portable, per_op_unknown


def classify_skip(reason: str) -> str:
    r = reason or ""
    if r.startswith("output decode") or "buffer length" in r or "client decode" in r or "non-tensor" in r:
        return "skip_feeder"
    return "skip_runtime"


def _mism_kind(reason: str) -> str:
    r = (reason or "").lower()
    if "non-finite" in r:
        return "nonfinite"
    if "shape" in r:
        return "shape"
    if "dtype" in r:
        return "dtype"
    return "value"


def main() -> int:
    if not SKIPLOG.exists():
        sys.exit(f"no skip-log at {SKIPLOG}")
    if not MANIFEST.exists():
        sys.exit(f"no manifest at {MANIFEST} (run build_manifest.py first)")

    j2op, j2deleg, per_op_total, per_op_deleg, per_op_portable, per_op_unknown = load_manifest()

    rows = []
    with open(SKIPLOG) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts += [""] * (4 - len(parts))
            status, job, reason, graph = parts[0], parts[1], parts[2], parts[3]
            rows.append((status, job, reason, graph))

    # per-op non-OK aggregation, split by delegated vs portable
    ops = collections.defaultdict(collections.Counter)
    op_reasons = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    buckets = {"mismatch_delegated": [], "mismatch_portable": [], "mismatch_unknown": [],
               "crash_delegated": [], "crash_portable": [], "crash_unknown": [],
               "skip_runtime": [], "skip_feeder": [], "timeout": []}

    for status, job, reason, graph in rows:
        op = j2op.get(job) or "?"
        d = j2deleg.get(job)
        tier = "unknown" if d is None else ("delegated" if d >= 1 else "portable")
        if status == "MISMATCH":
            key = f"mismatch_{tier}"
            buckets[key].append([op, job, reason, d])
            ops[op][key] += 1
            op_reasons[op]["MISMATCH"][_mism_kind(reason)] += 1
        elif status == "CRASH":
            key = f"crash_{tier}"
            buckets[key].append([op, job, reason, d])
            ops[op][key] += 1
            op_reasons[op]["CRASH"][reason[:80]] += 1
        elif status == "SKIP":
            sk = classify_skip(reason)
            buckets[sk].append([op, job, reason, d])
            ops[op][sk] += 1
            op_reasons[op]["SKIP"][reason[:80]] += 1
        elif status == "TIMEOUT":
            buckets["timeout"].append([op, job, reason, d])
            ops[op]["timeout"] += 1

    # per-op table (coverage ledger + worklist)
    with open(OUT / "per_op_table.tsv", "w") as f:
        f.write("operator\tk_samples\tdeleg\tportable\tunknown\tmism_deleg\tmism_port\t"
                "crash_deleg\tcrash_port\tskip_runtime\tskip_feeder\ttimeout\tOK_est\n")
        for op in sorted(per_op_total, key=lambda o: -per_op_total[o]):
            c = ops[op]
            nonok = (c["mismatch_delegated"] + c["mismatch_portable"] + c["mismatch_unknown"]
                     + c["crash_delegated"] + c["crash_portable"] + c["crash_unknown"]
                     + c["skip_runtime"] + c["skip_feeder"] + c["timeout"])
            ok_est = per_op_total[op] - nonok
            f.write(f"{op}\t{per_op_total[op]}\t{per_op_deleg[op]}\t{per_op_portable[op]}\t"
                    f"{per_op_unknown[op]}\t{c['mismatch_delegated']}\t{c['mismatch_portable']}\t"
                    f"{c['crash_delegated']}\t{c['crash_portable']}\t{c['skip_runtime']}\t"
                    f"{c['skip_feeder']}\t{c['timeout']}\t{ok_est}\n")

    with open(OUT / "buckets.json", "w") as f:
        json.dump(buckets, f, indent=1)

    # per-op reason strings (for skips.md / crash table)
    reason_dump = {op: {mode: dict(cnt) for mode, cnt in modes.items()}
                   for op, modes in op_reasons.items()}
    with open(OUT / "op_reasons.json", "w") as f:
        json.dump(reason_dump, f, indent=1)

    # coverage
    produced_ops = set(per_op_total)
    verdict_ops = set(op for op in ops)
    coverage = {
        "jobs_total": sum(per_op_total.values()),
        "ops_produced": len(produced_ops),
        "ops_with_nonok": len(verdict_ops),
        "nonok_rows": len(rows),
    }
    with open(OUT / "coverage.json", "w") as f:
        json.dump(coverage, f, indent=1)

    # summary
    tot = collections.Counter(r[0] for r in rows)
    print("=== corpus:", coverage["jobs_total"], "jobs;", coverage["ops_produced"], "ops produced")
    print("=== non-OK rows:", len(rows), dict(tot))
    for k, v in buckets.items():
        print(f"  {k:20s}: {len(v):6d}  ops={len(set(x[0] for x in v))}")

    print("\n=== MISMATCH on OpenVINO delegate (ops>=1) — by operator ===")
    md = collections.Counter(x[0] for x in buckets["mismatch_delegated"])
    for op, n in md.most_common():
        kinds = dict(op_reasons[op]["MISMATCH"])
        print(f"  {n:5d}  {op:40s} {kinds}")

    print("\n=== CRASH on OpenVINO delegate (ops>=1) — by operator ===")
    cd = collections.Counter(x[0] for x in buckets["crash_delegated"])
    for op, n in cd.most_common():
        print(f"  {n:5d}  {op}")

    print("\n=== SKIP (runtime reject) — by operator (top 40) ===")
    sk = collections.Counter(x[0] for x in buckets["skip_runtime"])
    for op, n in sk.most_common(40):
        print(f"  {n:5d}  {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
