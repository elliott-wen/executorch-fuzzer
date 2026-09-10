#!/usr/bin/env python3
"""Replay: aten::remainder integer path returns fmod sign (divisor-sign bug).

Usage: cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_remainder-sign.py
"""
import importlib.util as u
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")  # so `import mobile` resolves

import torch
from mobile.net import corpus, protocol, et_runner

REPO = "/data/jwen929/mobile"
JOB = "corpus/portable/w0/w0_452.py"   # remainder.Scalar(n13, -3), bool->int path
OUT = 1                                 # offending USER_OUTPUT index


def load(job_py):
    job = os.path.join(REPO, job_py.replace(".py", ".job"))
    frames = corpus.read_job(job)
    job_id, pte, inputs = protocol.decode_job(protocol.job_frames_from_pushjob(frames))
    return job_id, pte, inputs


def eager(job_py, inputs):
    s = u.spec_from_file_location("m", os.path.join(REPO, job_py))
    m = u.module_from_spec(s)
    s.loader.exec_module(m)
    outs = m.g(*[t.clone() for t in inputs])
    return [x for x in outs if hasattr(x, "dtype")]


def main():
    job_id, pte, inputs = load(JOB)
    pouts = et_runner.run_pte(pte, inputs)
    eouts = eager(JOB, inputs)
    p, e = pouts[OUT], eouts[OUT]
    pf, ef = p.to(torch.float64).flatten(), e.to(torch.float64).flatten()
    d = (pf - ef).abs()
    d = torch.where(torch.isnan(d), torch.zeros_like(d), d)
    md, mi = d.max(0)
    print(f"job {job_id} out[{OUT}] dt p={p.dtype}/e={e.dtype} "
          f"max|delta|={md.item():.4e} at idx {mi.item()}")
    print(f"  portable={p.flatten()[mi].item()!r}  eager={e.flatten()[mi].item()!r}")
    reproduced = md.item() > 0
    print("REPLAY: bug REPRODUCED" if reproduced else "REPLAY: no divergence")
    return 0 if reproduced else 1


if __name__ == "__main__":
    sys.exit(main())
