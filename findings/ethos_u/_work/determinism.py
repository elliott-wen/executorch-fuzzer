#!/usr/bin/env python3
"""Determinism gate (analysis_single.md step 3b): re-run each given job_id N times on the
SAME .pte and tally statuses. all-N repro -> bug; >=1/N -> INTERMITTENT; 0/N -> flaky/drop.

Runs each pass as a separate `mobile feed <job_id>` (debug mode) against the live broker on
15564/15566, so it reuses the graph's own inputs. Interleaves with the main feed.

Usage: determinism.py N job_id [job_id ...]
"""
import json, subprocess, sys, collections
from pathlib import Path

N = int(sys.argv[1])
job_ids = sys.argv[2:]
CORPUS = "mobile/corpus_v3/ethos-u"
CWD = "/data/jwen929"

def one_pass(jid):
    p = subprocess.run(
        ["python", "-m", "mobile", "feed", jid,
         "--corpus", CORPUS, "--host", "127.0.0.1",
         "--job-port", "15564", "--ctrl-port", "15566", "--timeout", "300"],
        cwd=CWD, capture_output=True, text=True, timeout=900)
    # debug mode prints one JSON object per graph
    for ln in p.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                r = json.loads(ln)
                if r.get("job_id") == jid:
                    return r.get("status", "?"), r.get("detail", ""), r
            except Exception:
                pass
    return "NORESULT", p.stdout[-200:], None

print(f"{'job_id':<14}{'op_chain':<34}{'N':>3}  statuses -> verdict")
for jid in job_ids:
    statuses = []
    detail0 = ""
    desc0 = ""
    for k in range(N):
        st, det, r = one_pass(jid)
        statuses.append(st)
        if r and not desc0:
            desc0 = r.get("op_chain", "")
        if st in ("MISMATCH", "CRASH", "SKIP") and not detail0:
            detail0 = det
    c = collections.Counter(statuses)
    nonok = sum(v for k, v in c.items() if k != "OK")
    if all(s != "OK" for s in statuses) and statuses[0] in ("MISMATCH", "CRASH", "SKIP"):
        verdict = f"REPRO {statuses[0]} (all {N})"
    elif nonok > 0:
        verdict = f"INTERMITTENT ({nonok}/{N} non-OK)"
    else:
        verdict = "FLAKY/clean (0 non-OK) -> drop"
    print(f"{jid:<14}{desc0[:33]:<34}{N:>3}  {dict(c)} -> {verdict}")
    if detail0:
        print(f"    detail: {detail0[:160]}")
