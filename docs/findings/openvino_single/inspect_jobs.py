#!/usr/bin/env python3
"""inspect_jobs.py — for given job_ids, load the stored inputs + eager oracle and report
finiteness of the leaf inputs vs eager (step 3e leaf check), plus shapes/dtypes. Also runs the
job on the live fleet to get the device output. Usage:
  PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/inspect_jobs.py w0:139 ...
"""
import sys, json, subprocess
sys.path.insert(0, "/data/jwen929")
from pathlib import Path
from mobile.executor import corpus as C
from mobile.executor import protocol as P
import torch

ROOT = Path("/data/jwen929/mobile")
CORPUS = str(ROOT / "corpus_v3/openvino")


def dev_run(job):
    cmd = [f"{ROOT}/.venv/bin/python", "-m", "mobile", "feed", "--corpus", CORPUS,
           "--job-port", "15664", "--ctrl-port", "15666", "--timeout", "60", job]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd="/data/jwen929",
                       env={"PYTHONPATH": "/data/jwen929", "PATH": f"{ROOT}/.venv/bin:/usr/bin:/bin"})
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {}


def fin(t):
    if not torch.is_tensor(t) or not t.is_floating_point():
        return "n/a(int)"
    return "FINITE" if torch.isfinite(t).all() else "HAS_NONFIN"


for job in sys.argv[1:]:
    path = C.job_file(CORPUS, job)
    frames = C.read_job(path)
    _jid, pte, inputs = P.decode_job(P.job_frames_from_pushjob(frames))
    eager, user_pos = P.eager_from_pushjob(frames)
    d = dev_run(job)
    oc = d.get("op_chain", "")
    print(f"\n=== {job}  {oc}")
    for i, t in enumerate(inputs):
        print(f"  leaf[{i}] {str(tuple(t.shape)):14s} {str(t.dtype):14s} {fin(t)}  "
              f"vals={t.flatten()[:6].tolist()}")
    for i, e in enumerate(eager):
        print(f"  eager[{i}] {str(tuple(e.shape)):14s} {str(e.dtype):14s} {fin(e)}  "
              f"vals={e.flatten()[:6].tolist() if torch.is_tensor(e) else e}")
    if d.get("outputs"):
        o = d["outputs"][0]
        print(f"  DEVICE status={d.get('status')} shape={o.get('et_shape', o.get('shape'))} "
              f"dev={str(o.get('et'))[:70]}")
    else:
        print(f"  DEVICE status={d.get('status')} detail={d.get('detail','')[:80]}")
