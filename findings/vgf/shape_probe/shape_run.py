#!/usr/bin/env python3
"""Decisive test: run the B-side (live-cone) graph of each SHAPE case on the VGF runtime and
compare the DEVICE output byte count against the eager reference's byte count.
device_bytes != eager_bytes  -> real runtime shape/IO-size divergence (plan was already proven correct)
device_bytes == eager_bytes  -> the SHAPE verdict was not a shape bug (harness/transient artifact)"""
import os, sys, subprocess, tempfile, pathlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.net import corpus as C, protocol as P
from mobile.gen.export import build_job
from mobile.gen.diff.cone_predicate import cone_localizer

CO = "/data/jwen929/mobile/corpus_v4/vgf"
RUNNER_SH = "/data/jwen929/mobile/vgf_client/vgf_runner.sh"
CASES = [
    ("w103:715", 3, "unfold_copy", ["n0","n4"]),   ("w104:305", 0, "neg", ["n0"]),
    ("w10:591",  1, "div", ["n1","n4"]),           ("w106:621", 0, "clamp", ["n2"]),
    ("w107:200", 0, "cumsum", ["n4","n3","n2","n1","n0"]),
    ("w107:319", 1, "log", ["n1","n7","n3"]),      ("w111:159", 1, "scatter_add", ["n0","n2"]),
    ("w114:256", 0, "min", ["n3","n0","n2"]),      ("w11:655", 0, "div", ["n2"]),
    ("w120:300", 2, "squeeze_copy", ["n0","n1","n5"]),
    ("w121:253", 1, "reciprocal", ["n2"]),         ("w22:81", 1, "view_copy", ["n0","n3"]),
]

def run_pte(j, tag):
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"shp_{tag}_"))
    pte = d / "m.pte"; pte.write_bytes(j.pte)
    cmd = [RUNNER_SH, "--pte", str(pte), "--out", str(d)]
    for i, t in enumerate(j.inputs):
        _, raw = P.tensor_to_meta_blob(t.contiguous())
        p = d / f"in_{i}.bin"; p.write_bytes(raw); cmd += ["--input", str(p)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    outs = sorted(d.glob("out_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    return r.returncode, [(p.name, p.stat().st_size) for p in outs], (d / "run.log")

for job_id, out_idx, tgt, needed in CASES:
    try:
        anc, build_src, target = cone_localizer(C.job_file(CO, job_id, "py"), out_idx)
        j = build_job(build_src(needed), "vgf", False)
    except Exception as e:
        print(f"{job_id:10s} {tgt:13s} BUILD_EXC {type(e).__name__}"); continue
    if j.status != "READY":
        print(f"{job_id:10s} {tgt:13s} BUILD:{j.status}"); continue
    e0 = j.eager[0]
    want_bytes = e0.numel() * e0.element_size()
    try:
        rc, outs, log = run_pte(j, job_id.replace(":", "_"))
    except Exception as e:
        print(f"{job_id:10s} {tgt:13s} RUN_EXC {type(e).__name__}"); continue
    up = j.user_pos
    pick = None
    if outs:
        idx = up[0] if (up and len(up) == 1 and up[0] < len(outs)) else len(outs) - 1
        pick = outs[idx]
    verdict = "?"
    if rc != 0:
        verdict = f"RUNNER_rc={rc}"
    elif pick is None:
        verdict = "NO_OUTPUT"
    elif pick[1] == want_bytes:
        verdict = "SIZE_MATCHES (not a shape bug)"
    else:
        verdict = f"*** SIZE DIFFERS: device={pick[1]}B vs eager={want_bytes}B ***"
    print(f"{job_id:10s} {tgt:13s} eager={tuple(e0.shape)} {e0.dtype} want={want_bytes}B "
          f"user_pos={up} device_outs={outs} -> {verdict}")
    if rc != 0:
        try: print("    log:", log.read_text()[-220:].replace("\n", " | "))
        except Exception: pass
