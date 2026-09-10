#!/usr/bin/env python3
"""nf_xb_one.py <backend> <jsonl> <lineno> — lower ONE reconstructed B-side graph to <backend>
and run it on that backend's local runtime, in a dedicated process (MOBILE_BACKENDS isolates).
Prints a single JSON line to stdout.  Native aborts kill only this process.
"""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

backend, jsonl, ln = sys.argv[1], sys.argv[2], int(sys.argv[3])
rec = json.loads(open(jsonl).read().splitlines()[ln])
src, pB = rec["_srcB"], rec["_pB"]
out = dict(job=rec["job"], out=rec["out"], backend=backend)

try:
    import torch
    from mobile.gen.export import build_job
    from mobile.net import compare as cmp
    from nf_common import nf_signature, classify
    j = build_job(src, backend, False)
    out["status"] = j.status
    out["deleg"] = json.dumps(getattr(j, "delegated", None) or {})
    if j.status != "READY":
        print(json.dumps(out)); sys.exit(0)
    from mobile.net.et_runner import run_pte
    raw = run_pte(j.pte, j.inputs)
    et = cmp.select(list(raw), j.user_pos)
    if len(et) > len(j.eager):
        et = et[len(et) - len(j.eager):]
    p = pB if pB >= 0 else len(j.eager) - 1
    ref, dev = j.eager[p], et[p]
    if not isinstance(dev, torch.Tensor):
        out["res"] = "NONTENSOR"; print(json.dumps(out)); sys.exit(0)
    st, mech, kind = classify(ref, dev)
    out["res"] = st; out["mech"] = mech; out["kind"] = kind
    out["ref_nf"] = json.dumps(nf_signature(ref))
    out["dev_nf"] = json.dumps(nf_signature(dev))
    out["ref8"] = [round(float(x), 5) for x in ref.flatten().float()[:8].tolist()]
    out["dev8"] = [round(float(x), 5) for x in dev.flatten().float()[:8].tolist()]
except BaseException as e:
    out["res"] = f"EXC:{type(e).__name__}:{str(e)[:100]}"
print(json.dumps(out))
