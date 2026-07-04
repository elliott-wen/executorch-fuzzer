#!/usr/bin/env python3
"""analysis_single.md step 3b — determinism gate.

Re-run each candidate job_id's EXACT .pte N>=5 times on the device and classify:
  CONFIRMED    - reproduces the SAME non-OK status in all N rounds
  INTERMITTENT - fails in 1..N-1 rounds
  FLAKY        - never reproduces (0/N) -> drop, do not file

Each round feeds the whole candidate set once (debug mode, one JSON per job), so rounds run
in parallel across the connected workers. Prints a per-job verdict table + JSON summary.

Usage: rerun.py <jobs.txt> <corpus_dir> [rounds=5] [timeout=60]
  jobs.txt: one job_id per line (e.g. w1:9)
"""
import sys, os, subprocess, json, collections

jobs_file, corpus = sys.argv[1], os.path.abspath(sys.argv[2])
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
TIMEOUT = int(sys.argv[4]) if len(sys.argv) > 4 else 60

job_ids = [l.strip() for l in open(jobs_file) if l.strip()]
def job_path(jid):
    shard, n = jid.split(":")
    return os.path.join(corpus, shard, f"{shard}_{n}.job")
paths = [job_path(j) for j in job_ids if os.path.exists(job_path(j))]

PY = "/data/jwen929/mobile/.venv/bin/python"
env = dict(os.environ, PYTHONPATH="/data/jwen929", CUDA_VISIBLE_DEVICES="")

# per job: list of statuses across rounds; keep first non-OK detail
rounds = collections.defaultdict(list)
detail = {}
for r in range(ROUNDS):
    cmd = [PY, "-m", "mobile", "feed", "--host", "127.0.0.1", "--job-port", "15574",
           "--ctrl-port", "15576", "--corpus", corpus, "--timeout", str(TIMEOUT)] + paths
    p = subprocess.run(cmd, cwd="/data/jwen929", env=env, capture_output=True, text=True,
                       timeout=TIMEOUT * len(paths) + 300)
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        jid, st = rec.get("job_id"), rec.get("status")
        rounds[jid].append(st)
        if st != "OK" and jid not in detail:
            detail[jid] = rec.get("detail", "")[:120]
    print(f"  round {r+1}/{ROUNDS} done ({len(rounds)} jobs seen)", file=sys.stderr)

def verdict(sts):
    nonok = [s for s in sts if s != "OK"]
    if not sts:
        return "NORESULT"
    if len(nonok) == len(sts):
        # same non-OK status every round?
        return "CONFIRMED" if len(set(nonok)) == 1 else "CONFIRMED-mixed"
    if nonok:
        return "INTERMITTENT"
    return "FLAKY"

summary = {}
print(f"{'job_id':14} {'verdict':16} {'statuses':30} detail")
for jid in job_ids:
    sts = rounds.get(jid, [])
    v = verdict(sts)
    summary[jid] = {"verdict": v, "statuses": sts, "detail": detail.get(jid, "")}
    print(f"{jid:14} {v:16} {','.join(sts):30} {detail.get(jid,'')}")

print("\n== verdict counts ==")
vc = collections.Counter(s["verdict"] for s in summary.values())
print(dict(vc))
json.dump(summary, open(os.path.join(os.path.dirname(jobs_file), "rerun_summary.json"), "w"), indent=1)
