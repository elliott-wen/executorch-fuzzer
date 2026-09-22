#!/usr/bin/env python3
"""Shared repro helper: replay exact corpus job(s) on the ENN device via the broker (port 15554)
and print eager vs device. Every job listed is a CONFIRMED (5/5 deterministic) delegated failure.

Usage: python _replay.py <job_id> [<job_id> ...]
Needs the broker up (feeder job-port 15554) and the Exynos E9965 worker connected.
"""
import os, sys, json, subprocess
CORPUS = "/data/jwen929/mobile/corpus_v3/samsung"
PY = "/data/jwen929/mobile/.venv/bin/python"

def path(jid):
    s, n = jid.split(":"); return f"{CORPUS}/{s}/{s}_{n}.job"

def replay(job_ids):
    paths = [path(j) for j in job_ids]
    env = dict(os.environ, PYTHONPATH="/data/jwen929", CUDA_VISIBLE_DEVICES="")
    cmd = [PY, "-m", "mobile", "feed", "--host", "127.0.0.1", "--job-port", "15554",
           "--ctrl-port", "15556", "--corpus", CORPUS, "--timeout", "60"] + paths
    p = subprocess.run(cmd, cwd="/data/jwen929", env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"): continue
        r = json.loads(line)
        print(f"\n{r['job_id']}  {r['op_chain']}  -> {r['status']}")
        if r.get("detail"): print("  detail:", r["detail"][:120])
        for o in r.get("outputs", []):
            print(f"  dtype={o['dtype']} shape={o['shape']}")
            print("  eager :", o.get("eager", [])[:8])
            print("  device:", o.get("et", [])[:8])

if __name__ == "__main__":
    replay(sys.argv[1:] or [])
