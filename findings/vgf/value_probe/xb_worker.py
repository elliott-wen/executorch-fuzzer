#!/usr/bin/env python3
"""xb_worker.py DUMP.pt BACKEND  — lower the reconstructed B graph to BACKEND and run it
IN-PROCESS via mobile.net.et_runner.run_pte (one backend per subprocess: a kernel failure is a
native abort). Prints a single JSON line."""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/data/jwen929")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.gen.export import build_job
from mobile.net import compare as cmp, et_runner

f, backend = sys.argv[1], sys.argv[2]
D = torch.load(f, weights_only=False)
res = {"file": os.path.basename(f), "backend": backend}
try:
    j = build_job(D["srcB"], backend, False)
    res["status"] = j.status
    res["deleg"] = int(getattr(j, "delegated_ops", -1))
    if j.status != "READY":
        print(json.dumps(res)); sys.exit(0)
    outs = list(et_runner.run_pte(j.pte, j.inputs))
    outs = cmp.select(outs, j.user_pos)
    if len(outs) > len(j.eager):
        outs = outs[len(outs) - len(j.eager):]
    pos = D["posB"]
    ref, dev = j.eager[pos], outs[pos]
    s, det = cmp._cmp(ref, dev, 0)
    res["verdict"] = s
    res["detail"] = det
    rf = ref.flatten().float(); df = dev.flatten().float()
    res["ref"] = [round(float(x), 6) for x in rf[:6]]
    res["dev"] = [round(float(x), 6) for x in df[:6]]
    fin = torch.isfinite(rf) & torch.isfinite(df)
    if bool(fin.any()):
        res["maxdelta"] = float((df[fin] - rf[fin]).abs().max())
        res["rel_sc"] = res["maxdelta"] / max(float(rf.abs().max()), 1e-12)
    # does the host device output equal the VGF device output?
    vd = D["dev"].flatten().float()
    if vd.numel() == df.numel():
        res["eq_vgf"] = bool(torch.equal(vd, df)) or bool(torch.allclose(vd, df, rtol=1e-3, atol=1e-4))
except Exception as e:
    res["status"] = f"EXC:{type(e).__name__}:{str(e)[:120]}"
print(json.dumps(res))
