"""Helper for the minimal buggy-graph files: lower a graph to the XNNPACK delegate, dispatch it
to the connected ARM phone through the broker, and print device-vs-eager.

These bugs only manifest on the ARM NEON microkernels, so the graph must run on the phone (the
x86 host computes them correctly). Prereq: `python broker.py` running + the ARM phone attached.
Each graph_*.py calls trigger(__file__, ...) from its __main__ block.
"""
import json
import os
import subprocess
import sys

REPO = "/data/jwen929/mobile"
sys.path.insert(0, "/data/jwen929")


def trigger(graph_path, job_id, desc=""):
    from mobile.gen.export import build_job
    from mobile.net import protocol as P, corpus as C

    src = open(graph_path).read()
    r = build_job(src, "xnnpack")                       # eager run + XNNPACK lowering → .pte
    if r.status != "READY":
        print(f"LOWER FAILED: {r.status} {getattr(r, 'reason', '')}")
        return 1
    cdir = os.path.join(REPO, "tmp/minrepro")
    frames = P.encode_pushjob(job_id, r.pte, r.inputs, r.eager, desc=desc, user_pos=r.user_pos)
    C.write_job(cdir, job_id, frames, src=src)
    out = subprocess.run([f"{REPO}/.venv/bin/python", "feed.py", job_id,
                          "--corpus", "tmp/minrepro", "--host", os.environ.get("BROKER_HOST", "127.0.0.1")],
                         cwd=REPO, capture_output=True, text=True, timeout=120).stdout
    d = next((json.loads(l) for l in out.splitlines() if l.startswith("{")), None)
    if d is None:
        print("no result — is the broker + ARM phone up?")
        return 1
    print(f"{job_id}: {d['status']} | {d.get('detail')}")
    for i, o in enumerate(d.get("outputs", [])):
        if str(o.get("eager")) != str(o.get("et")):
            print(f"  out[{i}] {o.get('dtype')}: eager={o.get('eager')}  DEVICE(ARM)={o.get('et')}")
    ok = d["status"] in ("MISMATCH", "CRASH")
    print("=> bug REPRODUCED on ARM" if ok else "=> no divergence")
    return 0 if ok else 1
