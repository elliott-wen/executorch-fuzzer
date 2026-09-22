#!/usr/bin/env python3
"""For each job_id: print leaves (dtype+vals+finite), eager out, and device out (via debug feed).
Resolves mechanism (WRONG-VALUE/SCALE/ZEROED/dtype) for the FINITE-CLEAN mismatch worklist."""
import sys, runpy, json, subprocess
import torch

CORPUS = "/data/jwen929/mobile/corpus_v3/ethos-u"
CWD = "/data/jwen929"

def py(jid):
    w, n = jid.split(":")
    return f"{CORPUS}/{w}/{w}_{n}.py"

def device_out(jid):
    p = subprocess.run(["python", "-m", "mobile", "feed", jid, "--corpus", "mobile/corpus_v3/ethos-u",
                        "--host", "127.0.0.1", "--job-port", "15564", "--ctrl-port", "15566", "--timeout", "200"],
                       cwd=CWD, capture_output=True, text=True, timeout=400)
    for ln in p.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                r = json.loads(ln)
                if r.get("job_id") == jid:
                    return r.get("status"), r.get("detail"), r.get("outputs")
            except Exception:
                pass
    return "?", p.stdout[-150:], None

for jid in sys.argv[1:]:
    print(f"\n===== {jid} =====")
    ns = runpy.run_path(py(jid))
    L = ns["LEAVES"]; g = ns["g"]
    # graph line
    for ln in open(py(jid)):
        if ln.startswith("Graph:") or "Graph:" in ln:
            print("  " + ln.strip()); break
    for i, t in enumerate(L):
        fin = bool(torch.isfinite(t.float()).all()) if t.numel() else True
        print(f"  L{i} {t.dtype} {tuple(t.shape)} finite={fin} = {t.flatten().tolist()[:10]}")
    o = g(*L); o = o[0] if isinstance(o, (tuple, list)) else o
    print(f"  EAGER {o.dtype} {tuple(o.shape)} = {o.flatten().tolist()[:12]}")
    st, det, outs = device_out(jid)
    print(f"  DEVICE status={st} detail={det}")
    if outs:
        for x in (outs if isinstance(outs, list) else [outs]):
            print(f"    dev dtype={x.get('dtype')} shape={x.get('shape')} et={x.get('et')[:12] if x.get('et') else x.get('et')}")
            print(f"        eager(from oracle)={x.get('eager')[:12] if x.get('eager') else x.get('eager')}")
