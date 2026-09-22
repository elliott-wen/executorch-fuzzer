#!/usr/bin/env python3
"""Shared repro driver for the cortex-m single-operator findings.

Each repro_<op>.py sets JOB_ID and calls run(JOB_ID). This:
  1. loads the graph's own LEAVES + fp32 eager (the exact inputs the corpus used),
  2. feeds the pre-lowered one-op .pte to a live FVP cortex-m worker via the broker,
  3. prints the eager (stored quantized reference) vs device (Cortex-M55 CMSIS-NN on the
     Corstone FVP) values side by side, plus the device status/detail.

cortex-m is PASS-BASED, not a delegate: the CortexM pass rewrites the quantized graph into
`cortex_m::` CMSIS-NN custom ops (quantize/dequantize boundary on every quantized graph, plus
a few compute kernels: transpose/minimum/quantized_mul/pad/…). So `delegated ops` is always 0;
the "which kernel ran" signal is which `cortex_m::` ops the .pte calls (see _work/cm_kernels.py).

PREREQUISITE — a broker + at least one FVP cortex-m client must be running, e.g.:
    cd /data/jwen929 && python -m mobile broker --job-port 15664 --client-port 15665 --ctrl-port 15666 &
    cd /data/jwen929 && python mobile/fvp_client/fvp_client.py --host 127.0.0.1 \
        --client-port 15665 --ctrl-port 15666 --backend cortex-m --target cortex-m55 &
(the exact fleet this corpus was fed through; see findings/cortex_m/_work/start_broker.sh + start_fleet.sh)

Every graph is ONE operator wired to leaf inputs (corpus generated with --nodes 1), int8
(quant=ALWAYS), so the divergence is attributable to that one operator's lowering + kernel.
"""
import os, sys, json, runpy, subprocess
import torch

CORPUS = "mobile/corpus_v3/cortex-m"
CWD = "/data/jwen929"
JOB_PORT = os.environ.get("JOB_PORT", "15664")
CTRL_PORT = os.environ.get("CTRL_PORT", "15666")


def _py(jid):
    w, n = jid.split(":")
    return f"{CWD}/{CORPUS}/{w}/{w}_{n}.py"


def run(job_id):
    src = _py(job_id)
    ns = runpy.run_path(src)
    L, g = ns["LEAVES"], ns["g"]
    graph = next((ln.strip() for ln in open(src) if "Graph:" in ln), job_id)

    print(f"# repro {job_id}: {graph}")
    for i, t in enumerate(L):
        fin = bool(torch.isfinite(t.float()).all()) if t.numel() else True
        print(f"#   L{i} {t.dtype} {tuple(t.shape)} finite={fin} = {t.flatten().tolist()[:12]}")
    o = g(*L)
    o = o[0] if isinstance(o, (tuple, list)) else o
    print(f"#   eager(fp32) {o.dtype} {tuple(o.shape)} = {o.flatten().tolist()[:16]}")

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
        print("!! no device result — is a broker+FVP cortex-m client running? stdout tail:")
        print(p.stdout[-400:] or p.stderr[-400:])
        return
    print(f"\nDEVICE status={rec.get('status')}  detail={rec.get('detail')}")
    for x in (rec.get("outputs") or []):
        print(f"  eager(quant-ref) = {x.get('eager')}")
        print(f"  device           = {x.get('et')}   (dtype {x.get('dtype')} shape {x.get('shape')})")
