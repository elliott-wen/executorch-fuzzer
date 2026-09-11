#!/usr/bin/env python3
"""Replay: aliased out= resize divergence (behavioral execution-model difference).

ATen out-variants (topk/min.dim/max.dim) resize their aliased out= buffer in place
at runtime; the new shape propagates to downstream nodes. ExecuTorch portable uses
statically-planned buffers, never resizes, so the aliased tensor keeps its original
broadcast_to(...,[0,0,0,0,0]) shape and portable SILENTLY returns the wrong shape.

Comparison is on SHAPE, not value: reproduced iff portable_out.shape != eager_out.shape.

Usage: cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_shape-aliased-out-resize.py
"""
import importlib.util as u
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")  # so `import mobile` resolves

import torch
from mobile.executor import corpus, protocol, et_runner

REPO = "/data/jwen929/mobile"
JOB = "corpus/portable/w0/w0_1246.py"   # topk.values aliasing broadcast_to(...,[0,0,0,0,0])
OUT = 0                                  # offending USER_OUTPUT index


def load(job_py):
    job = os.path.join(REPO, job_py.replace(".py", ".job"))
    frames = corpus.read_job(job)
    job_id, pte, inputs = protocol.decode_job(protocol.job_frames_from_pushjob(frames))
    return job_id, pte, inputs


def eager(job_py, inputs):
    s = u.spec_from_file_location("m", os.path.join(REPO, job_py))
    m = u.module_from_spec(s)
    s.loader.exec_module(m)
    # m.g executes nodes in graph order, so the topk in-place resize propagates.
    outs = m.g(*[t.clone() for t in inputs])
    return [x for x in outs if hasattr(x, "shape")]


def main():
    job_id, pte, inputs = load(JOB)
    pouts = et_runner.run_pte(pte, inputs)
    eouts = eager(JOB, inputs)
    p, e = pouts[OUT], eouts[OUT]
    ps, es = tuple(p.shape), tuple(e.shape)
    print(f"job {job_id} out[{OUT}] SHAPE portable={ps}  eager={es}")
    reproduced = ps != es
    print("REPLAY: bug REPRODUCED (shapes differ)" if reproduced
          else "REPLAY: no shape divergence")
    return 0 if reproduced else 1


if __name__ == "__main__":
    sys.exit(main())
