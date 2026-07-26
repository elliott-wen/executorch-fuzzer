#!/usr/bin/env python3
"""determinism_gate.py — step 3b of analysis_single.md for OpenVINO.

Re-run each delegated failing job (MISMATCH/CRASH, ops>=1) N times through the LIVE broker+client
fleet and classify:
  REPRO        = all N re-runs reproduce a non-OK verdict  -> keep as a bug
  INTERMITTENT = 1..N-1 re-runs fail                        -> report as intermittent
  FLAKY        = 0 re-runs fail (all OK)                    -> drop

The feeder collapses same-job repeats within one invocation, so N determinism requires N SEPARATE
feed processes (each runs every candidate once). Rich-JSON output gives per-job status + mechanism.

Run: PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/determinism_gate.py [N] [MAX_PER_OP]
"""
from __future__ import annotations
import sys, json, subprocess, collections
from pathlib import Path

ROOT = Path("/data/jwen929/mobile")
OUT = ROOT / "findings/openvino_single"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MAX_PER_OP = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def mechanism(o) -> str:
    st = o.get("status")
    if st != "MISMATCH":
        return st or "?"
    outs = o.get("outputs") or []
    if not outs:
        return "value"
    out = outs[0]
    eg, et = out.get("eager"), out.get("et")
    if "et_shape" in out and out.get("et_shape") != out.get("shape"):
        return "shape"
    if et is not None and eg is not None:
        try:
            if all(v == 0 for v in et) and any(v != 0 for v in eg):
                return "ZEROED"
        except Exception:
            pass
    if "non-finite" in (o.get("detail") or ""):
        return "nonfinite"
    return "value"


def run_feed(job_ids):
    """One feed invocation over job_ids -> {job_id: (status, mechanism, detail)}."""
    cmd = [f"{ROOT}/.venv/bin/python", "-m", "mobile", "feed", "--corpus",
           f"{ROOT}/corpus_v3/openvino", "--job-port", "15664", "--ctrl-port", "15666",
           "--timeout", "60"] + job_ids
    p = subprocess.run(cmd, capture_output=True, text=True, cwd="/data/jwen929",
                       env={"PYTHONPATH": "/data/jwen929",
                            "PATH": f"{ROOT}/.venv/bin:/usr/bin:/bin"})
    res = {}
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        res[o["job_id"]] = (o.get("status"), mechanism(o), (o.get("detail") or "")[:70])
    return res


def main():
    buckets = json.loads((OUT / "buckets.json").read_text())
    cand = []          # (op, bucket, job)
    seen = set()
    for bkey in ("mismatch_delegated", "crash_delegated"):
        by_op = collections.defaultdict(list)
        for op, job, reason, d in buckets.get(bkey, []):
            by_op[op].append(job)
        for op, jobs in by_op.items():
            for j in jobs[:MAX_PER_OP]:
                if j not in seen:
                    seen.add(j)
                    cand.append((op, bkey, j))
    job_ids = [c[2] for c in cand]
    print(f"determinism gate: {len(job_ids)} jobs "
          f"({len(set(c[0] for c in cand))} ops), N={N} separate feeds", flush=True)

    runs = []
    for i in range(N):
        r = run_feed(job_ids)
        runs.append(r)
        print(f"  feed {i+1}/{N}: {len(r)} results", flush=True)

    # classify per job
    records = []
    for op, bkey, j in cand:
        verdicts = [runs[i].get(j) for i in range(N)]
        got = [v for v in verdicts if v is not None]
        non_ok = [v for v in got if v[0] not in ("OK", None)]
        if not got:
            cls = "NO_RESULT"
        elif len(non_ok) == len(got) and len(got) == N:
            cls = "REPRO"
        elif len(non_ok) == 0:
            cls = "FLAKY"
        else:
            cls = "INTERMITTENT"
        mechs = collections.Counter(v[1] for v in non_ok)
        records.append({"op": op, "bucket": bkey, "job": j, "class": cls,
                        "n": N, "got": len(got), "fail": len(non_ok),
                        "mech": (mechs.most_common(1)[0][0] if mechs else ""),
                        "detail": (non_ok[0][2] if non_ok else "")})

    (OUT / "determinism.json").write_text(json.dumps(records, indent=1))

    # per-op rollup
    op_cls = collections.defaultdict(collections.Counter)
    op_mech = collections.defaultdict(collections.Counter)
    for r in records:
        op_cls[(r["op"], r["bucket"])][r["class"]] += 1
        if r["class"] == "REPRO":
            op_mech[r["op"]][r["mech"]] += 1
    print(f"\n{'operator':40s} {'bucket':18s} REPRO INTERM FLAKY  mech(REPRO)")
    for (op, bkey), c in sorted(op_cls.items(), key=lambda kv: -kv[1]['REPRO']):
        print(f"{op:40s} {bkey:18s} {c['REPRO']:5d} {c['INTERMITTENT']:6d} {c['FLAKY']:5d}  {dict(op_mech[op])}")
    nrepro = sum(1 for r in records if r["class"] == "REPRO")
    print(f"\nREPRO jobs: {nrepro}/{len(records)}")


if __name__ == "__main__":
    main()
