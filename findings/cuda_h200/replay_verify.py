#!/usr/bin/env python3
"""replay_verify.py — determinism gate (3b) + value pinning for selected job_ids.

Re-runs each job_id N times through the running broker (needs serve_cuda.sh up), parses the rich
JSON the feeder emits per job, and reports: the multiset of verdicts over N runs (all-N same →
deterministic; mixed → INTERMITTENT; all-OK → flaky/drop), plus a compact value sample (max|delta|
or nan/inf, and device-vs-eager for the first differing output) so the mechanism can be named.

Usage: replay_verify.py [--n 5] <job_id> [<job_id> ...]
Prints one summary line per job_id and writes replay_verify.tsv.
"""
import json, subprocess, sys, argparse
from collections import Counter
from pathlib import Path

REPO = "/data/jwen929"; MOBILE = "/data/jwen929/mobile"
OUT = Path(MOBILE) / "findings/cuda_h200"

def run_feed(job_ids):
    """Feed explicit job_ids once each (rich JSON per job). Returns list of parsed records."""
    cmd = [f"{MOBILE}/.venv/bin/python", "-m", "mobile", "feed", *job_ids,
           "--corpus", f"{MOBILE}/corpus_v3/cuda", "--timeout", "120"]
    env = {"PYTHONPATH": REPO, "PATH": "/usr/bin:/bin"}
    import os
    e = {**os.environ, **env}
    p = subprocess.run(cmd, cwd=REPO, env=e, capture_output=True, text=True, timeout=1200)
    recs = []
    for ln in p.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try: recs.append(json.loads(ln))
            except Exception: pass
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("ids", nargs="+")
    a = ap.parse_args()
    # The feeder de-dupes a work list, so repeat the WHOLE batch N times (N separate feed passes)
    # to get N independent device runs of each job — the determinism gate.
    by = {}
    for _ in range(a.n):
        for r in run_feed(list(a.ids)):
            by.setdefault(r["job_id"], []).append(r)
    rows = []
    for jid in a.ids:
        rs = by.get(jid, [])
        verds = Counter(r["status"] for r in rs)
        got = sum(verds.values())
        # determinism verdict
        nonok = sum(v for k, v in verds.items() if k != "OK")
        if got == 0:
            det = "NO-RESULT"
        elif len(verds) == 1 and "OK" not in verds:
            det = f"DETERMINISTIC({list(verds)[0]} {got}/{got})"
        elif nonok == 0:
            det = f"FLAKY-DROP(OK {got}/{got})"
        elif nonok == got:
            det = f"DETERMINISTIC-nonOK({dict(verds)})"
        else:
            det = f"INTERMITTENT({dict(verds)})"
        # a value sample from the first non-OK rec with outputs/detail
        sample = ""
        for r in rs:
            if r["status"] != "OK":
                sample = (r.get("detail") or "")[:120]
                break
        rows.append((jid, det, verds.get("OK", 0), nonok, got, sample))
        print(f"{jid:14s} {det:36s} ok={verds.get('OK',0)} nonok={nonok}/{got}  {sample}")
    with open(OUT / "replay_verify.tsv", "w") as f:
        f.write("job_id\tdeterminism\tok\tnonok\truns\tsample\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")

if __name__ == "__main__":
    main()
