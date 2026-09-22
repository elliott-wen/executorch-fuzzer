#!/usr/bin/env python3
"""replay.py — feed explicit job_ids through the live broker, print device vs eager per output,
and flag ZEROED (device all-zero where eager is not). Usage:
  PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/replay.py w0:260 w0:263 ...
"""
import sys, json, subprocess
ROOT = "/data/jwen929/mobile"
jobs = sys.argv[1:]
cmd = [f"{ROOT}/.venv/bin/python", "-m", "mobile", "feed", "--corpus",
       f"{ROOT}/corpus_v3/openvino", "--job-port", "15664", "--ctrl-port", "15666"] + jobs
p = subprocess.run(cmd, capture_output=True, text=True, cwd="/data/jwen929",
                   env={"PYTHONPATH": "/data/jwen929", "PATH": f"{ROOT}/.venv/bin:/usr/bin:/bin"})
for line in p.stdout.splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        o = json.loads(line)
    except Exception:
        continue
    st = o.get("status")
    oc = o.get("op_chain", "")[:42]
    outs = o.get("outputs", [])
    if not outs:
        print(f"{o['job_id']:10s} {st:8s} {oc}")
        continue
    out = outs[0]
    eg = out.get("eager"); et = out.get("et")
    dt = out.get("dtype", "?")
    tag = ""
    if et is not None and eg is not None:
        try:
            zeroed = all(v == 0 for v in et) and any(v != 0 for v in eg)
            tag = "ZEROED" if zeroed else "value"
        except Exception:
            tag = ""
    elif "et_shape" in out:
        tag = f"shape {out.get('shape')} vs {out.get('et_shape')}"
    print(f"{o['job_id']:10s} {st:8s} {dt:8s} {tag:8s} {oc}")
    print(f"           eager={str(eg)[:60]}")
    print(f"           dev  ={str(et)[:60]}")
