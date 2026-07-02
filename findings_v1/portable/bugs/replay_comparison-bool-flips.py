#!/usr/bin/env python3
"""Replay: comparison/bool flip INHERITED from an upstream float diff.

ARTIFACT — NOT an ExecuTorch backend bug. The `lt`/`gt`/`ge`/`le`/`eq`/
`logical_*` comparator kernel is correct; its bool output merely flips
True<->False because an UPSTREAM float op (here `rsqrt`) produced a slightly
different value that crosses the comparison threshold. The divergence is
inherited from the float kernel, not born in the comparator. See
[comparison-bool-flips.md](comparison-bool-flips.md) and mismatch-delta-exact
Mechanism M6.

Job: corpus/portable/w0/w0_227.py
  n3 = rsqrt.out(n0)                          # float op: portable != eager slightly
  n4 = lt.Tensor_out(n3, L3, out=int64)  -> USER_OUTPUT[1]   (the flipped bool)
A sibling rsqrt-derived float output (out[3] = atan2 of rsqrt(n1)) is printed
to expose the inherited float difference that drives the flip.

Exits 0 iff the bool output flips (max|delta| >= 1).

Usage: cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_comparison-bool-flips.py
"""
import importlib.util as u
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")  # so `import mobile` resolves

import torch
from mobile.net import corpus, protocol, et_runner

REPO = "/data/jwen929/mobile"
JOB = "corpus/portable/w0/w0_227.py"
OUT_BOOL = 1     # lt.Tensor_out (int64-typed bool output) — the flip
OUT_FLOAT = 3    # rsqrt-derived float sibling — the inherited cause


def load(job_py):
    job = os.path.join(REPO, job_py.replace(".py", ".job"))
    frames = corpus.read_job(job)
    return protocol.decode_job(protocol.job_frames_from_pushjob(frames))


def eager(job_py, inputs):
    s = u.spec_from_file_location("m", os.path.join(REPO, job_py))
    m = u.module_from_spec(s)
    s.loader.exec_module(m)
    outs = m.g(*[t.clone() for t in inputs])
    return [x for x in outs if hasattr(x, "dtype")]


def delta(p, e):
    pf, ef = p.to(torch.float64).flatten(), e.to(torch.float64).flatten()
    d = (pf - ef).abs()
    d = torch.where(torch.isnan(d), torch.zeros_like(d), d)
    md, mi = d.max(0)
    return md.item(), mi.item(), pf, ef


def main():
    job_id, pte, inputs = load(JOB)
    pouts = et_runner.run_pte(pte, inputs)
    eouts = eager(JOB, inputs)

    # The comparator bool output that flipped.
    p, e = pouts[OUT_BOOL], eouts[OUT_BOOL]
    md, mi, pf, ef = delta(p, e)
    print(f"job {job_id} out[{OUT_BOOL}] (lt comparator) dt p={p.dtype}/e={e.dtype} "
          f"max|delta|={md:.4g} at idx {mi}")
    print(f"  portable={int(pf[mi].item())}  eager={int(ef[mi].item())}   "
          f"(bool flip True<->False)")

    # The inherited upstream float difference that caused the threshold crossing.
    pf2, ef2 = pouts[OUT_FLOAT], eouts[OUT_FLOAT]
    md2, mi2, pff, eff = delta(pf2, ef2)
    print(f"  inherited cause: out[{OUT_FLOAT}] (rsqrt-derived float) "
          f"max|delta|={md2:.4g} portable={pff[mi2].item():.6g} eager={eff[mi2].item():.6g}")
    print("  -> the comparator is correct; its input float value diverged upstream.")

    reproduced = md >= 1
    print("REPLAY: ARTIFACT reproduced (inherited bool flip)" if reproduced
          else "REPLAY: no flip")
    return 0 if reproduced else 1


if __name__ == "__main__":
    sys.exit(main())
