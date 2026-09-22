#!/usr/bin/env python3
"""Shared repro driver for the ethos-u single-operator findings.

Each repro_<op>.py sets JOB_ID and calls run(JOB_ID). This:
  1. loads the graph's own LEAVES + eager reference (the exact inputs the corpus used),
  2. feeds the pre-lowered one-op .pte to a live FVP ethos-u worker via the broker,
  3. prints eager (quantized reference) vs device (Ethos-U on Corstone FVP) side by side.

PREREQUISITE — a broker + at least one FVP ethos-u client must be running, e.g.:
    cd /data/jwen929 && python -m mobile broker --job-port 15564 --client-port 15565 --ctrl-port 15566 &
    cd /data/jwen929 && python mobile/fvp_client/fvp_client.py --host 127.0.0.1 \
        --client-port 15565 --ctrl-port 15566 --backend ethos-u --target ethos-u55-128 &
(the exact fleet this corpus was fed through; see findings/ethos_u/_work/start_broker.sh + start_fleet.sh)

Every graph is ONE operator wired to leaf inputs (corpus generated with --nodes 1), delegated
to the Ethos-U NPU (delegated ops>=1), so the divergence is attributable to that one kernel.
"""
import os, sys, json, runpy, subprocess
import torch

CORPUS = "mobile/corpus_v3/ethos-u"
CWD = "/data/jwen929"
JOB_PORT = os.environ.get("JOB_PORT", "15564")
CTRL_PORT = os.environ.get("CTRL_PORT", "15566")


def _py(jid):
    w, n = jid.split(":")
    return f"{CWD}/{CORPUS}/{w}/{w}_{n}.py"


def run(job_id):
    src = _py(job_id)
    ns = runpy.run_path(src)
    L, g = ns["LEAVES"], ns["g"]
    graph = next((ln.strip() for ln in open(src) if "Graph:" in ln), job_id)
    hdr = next((ln.strip() for ln in open(src) if ln.startswith("# delegated:")), "")

    print(f"# repro {job_id}: {graph}")
    print(f"# {hdr}")
    for i, t in enumerate(L):
        fin = bool(torch.isfinite(t.float()).all()) if t.numel() else True
        print(f"#   L{i} {t.dtype} {tuple(t.shape)} finite={fin} = {t.flatten().tolist()[:12]}")
    o = g(*L)
    o = o[0] if isinstance(o, (tuple, list)) else o
    print(f"#   eager {o.dtype} {tuple(o.shape)} = {o.flatten().tolist()[:16]}")

    p = subprocess.run(
        ["python", "-m", "mobile", "feed", job_id, "--corpus", CORPUS, "--host", "127.0.0.1",
         "--job-port", JOB_PORT, "--ctrl-port", CTRL_PORT, "--timeout", "200"],
        cwd=CWD, capture_output=True, text=True, timeout=400)
    rec = None
    for ln in p.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                r = json.loads(ln)
                if r.get("job_id") == job_id:
                    rec = r
            except Exception:
                pass
    if rec is None:
        print("!! no device result — is a broker+FVP client running? stdout tail:")
        print(p.stdout[-400:] or p.stderr[-400:])
        return
    print(f"\nDEVICE status={rec.get('status')}  detail={rec.get('detail')}")
    for x in (rec.get("outputs") or []):
        print(f"  eager  = {x.get('eager')}")
        print(f"  device = {x.get('et')}   (dtype {x.get('dtype')} shape {x.get('shape')})")
