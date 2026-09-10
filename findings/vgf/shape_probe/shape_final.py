#!/usr/bin/env python3
"""Replicate the A/B harness comparison for the 12 SHAPE cases: run B-side on VGF, parse the
USER_OUTPUT back into a tensor, and report the real verdict + mechanism now."""
import os, sys, subprocess, tempfile, pathlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.net import corpus as C, protocol as P, compare as cmp
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
NREP = 3

def run_once(j, tag):
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"sf_{tag}_"))
    pte = d / "m.pte"; pte.write_bytes(j.pte)
    cmd = [RUNNER_SH, "--pte", str(pte), "--out", str(d)]
    for i, t in enumerate(j.inputs):
        _, raw = P.tensor_to_meta_blob(t.contiguous())
        p = d / f"in_{i}.bin"; p.write_bytes(raw); cmd += ["--input", str(p)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0: return None, f"rc={r.returncode}"
    outs = sorted(d.glob("out_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    return outs, None

for job_id, out_idx, tgt, needed in CASES:
    try:
        anc, build_src, target = cone_localizer(C.job_file(CO, job_id, "py"), out_idx)
    except Exception as e:
        print(f"{job_id:10s} {tgt:13s} CONE_EXC"); continue
    verdicts = []
    for rep in range(NREP):
        try:
            j = build_job(build_src(needed), "vgf", False)
            if j.status != "READY": verdicts.append(f"BUILD:{j.status}"); continue
            outs, err = run_once(j, job_id.replace(":", "_"))
            if outs is None: verdicts.append(err); continue
            e0 = j.eager[0]
            up = j.user_pos
            idx = up[0] if (up and len(up) == 1 and up[0] < len(outs)) else len(outs) - 1
            buf = outs[idx].read_bytes()
            dev = torch.frombuffer(bytearray(buf), dtype=e0.dtype)
            if dev.numel() != e0.numel():
                verdicts.append(f"SHAPE(dev_numel={dev.numel()} vs {e0.numel()})"); continue
            dev = dev.reshape(e0.shape)
            st, det = cmp._cmp(e0, dev, 0)
            if st != "MISMATCH":
                verdicts.append("OK")
            else:
                r = e0.flatten().float(); dd = dev.flatten().float()
                if not torch.equal(r.isnan(), dd.isnan()): m = "NONFINITE"
                elif dd.abs().max() < 1e-6 and r.abs().max() > 1e-3: m = "ZEROED"
                else: m = "VALUE"
                verdicts.append(f"MISMATCH/{m}")
        except Exception as e:
            verdicts.append(f"EXC:{type(e).__name__}")
    print(f"{job_id:10s} {tgt:13s} eager={tuple(j.eager[0].shape)} {str(j.eager[0].dtype).replace('torch.','')} -> {verdicts}")
