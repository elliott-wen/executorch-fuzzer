#!/usr/bin/env python3
"""_replay.py — shared CoreML repro helper.

Replays one or more corpus jobs on the connected CoreML device by streaming them
through the running broker (exactly how the analysis verified them), and returns
the eager CPU reference vs the device (Core ML) output for each.

Contract (a human sets this up — see ../README.md):
  * broker up:  python -m mobile broker --job-port 15554 --client-port 15555 --ctrl-port 15556
  * a CoreML worker (Apple-Silicon Mac) connected on client-port 15555 and pulling.

The device only *executes* the stored .pte — no CoreML AoT env is needed here (the corpus was
lowered at generation time). Deterministic: every bug below reproduced 5/5 on re-run.

Usage from a repro:  from _replay import replay; replay(["w100:421", ...])
"""
import json, os, subprocess, sys

CORPUS = os.environ.get("COREML_CORPUS", "/data/jwen929/mobile/corpus_v3/coreml")
PY = os.environ.get("MOBILE_PY", "/data/jwen929/mobile/.venv/bin/python")


def replay(job_ids, timeout=30):
    """Feed job_ids through the broker once each; return {job_id: result-dict}."""
    cmd = [PY, "-m", "mobile", "feed", "--host", "127.0.0.1", "--job-port", "15554",
           "--ctrl-port", "15556", "--corpus", CORPUS, "--window", "1",
           "--timeout", str(timeout)] + list(job_ids)
    env = dict(os.environ, PYTHONPATH="/data/jwen929", CUDA_VISIBLE_DEVICES="")
    out = subprocess.run(cmd, cwd="/data/jwen929/mobile", capture_output=True,
                         text=True, env=env, timeout=timeout * len(job_ids) + 120).stdout
    res = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                o = json.loads(line); res[o["job_id"]] = o
            except Exception:
                pass
    return res


def show(job_ids, note=""):
    """Print eager-vs-device for each job. Returns the results dict."""
    res = replay(job_ids)
    if note:
        print(note)
    for j in job_ids:
        o = res.get(j)
        if not o:
            print(f"  {j}: NO RESULT (worker connected? broker up?)"); continue
        st = o["status"]
        outs = o.get("outputs") or [{}]
        u = outs[0]
        eg = u.get("eager"); et = u.get("et")
        head = f"  {j}: status={st}  op={o.get('op_chain','')}"
        if st == "MISMATCH":
            print(f"{head}\n      detail : {o.get('detail','')}\n"
                  f"      eager  : {eg}\n      device : {et}")
        elif st in ("CRASH", "TIMEOUT", "SKIP"):
            print(f"{head}\n      detail : {o.get('detail','')}")
        else:
            print(f"{head}  (ran OK — not reproduced this pass)")
    return res


if __name__ == "__main__":
    show(sys.argv[1:] or ["w100:421"])
