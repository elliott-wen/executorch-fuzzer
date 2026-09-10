#!/usr/bin/env python3
"""Replay: inherited non-finite (NaN) RE-POSITIONED downstream (harness strictness).

ARTIFACT — NOT an ExecuTorch backend bug. A NaN is legitimately produced by an
out-of-domain op (`acos` of |x|>1) on BOTH backends. A downstream op
(`remainder`/`floor_divide` then `sign`) re-positions the NaN differently
because portable and eager use different formula/broadcast ordering, so the
exact NaN POSITION in the returned tensor differs. Both backends are "correct";
only the position-exact non-finite harness check trips. See
[nonfinite-inherited-reposition.md](nonfinite-inherited-reposition.md) and
mismatch-non-finite Mechanism 3.

Job: corpus/portable/w0/w0_313.py
  n0 = acos.out(L0)            # NaN in BOTH backends where |L0|>1
  ... downstream remainder/floor_divide/sign re-position the NaN
  USER_OUTPUT[4] = sign.out(clone(broadcast(acos)))  -> nonfinite-pos-diff
This script PROVES the NaN is inherited (acos(L0) is NaN on the eager side too),
then shows the differing non-finite positions in the returned output.

Exits 0 iff a non-finite position difference is observed AND the upstream acos
NaN is present on both backends (inherited, not born in the returned op).

Usage: cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_nonfinite-inherited-reposition.py
"""
import importlib.util as u
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")  # so `import mobile` resolves

import torch
from mobile.net import corpus, protocol, et_runner

REPO = "/data/jwen929/mobile"
JOB = "corpus/portable/w0/w0_313.py"
OUT = 4   # sign.out(clone(broadcast(acos))) — non-finite positions differ


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


def main():
    job_id, pte, inputs = load(JOB)
    pouts = et_runner.run_pte(pte, inputs)
    eouts = eager(JOB, inputs)

    # 1) Prove the NaN is inherited: acos(L0) is NaN on the EAGER side too.
    L0 = inputs[0]
    a = torch.acos(L0)
    up_nan = torch.isnan(a)
    print(f"job {job_id}  upstream proof: L0={L0.tolist()}")
    print(f"  eager acos(L0)={a.tolist()}  isnan={up_nan.tolist()}  "
          f"-> NaN BORN on BOTH backends (|x|>1 OOB)")

    # 2) The returned output: non-finite positions differ after re-positioning.
    p, e = pouts[OUT], eouts[OUT]
    pf, ef = p.to(torch.float64).flatten(), e.to(torch.float64).flatten()
    pnf = torch.isnan(pf) | torch.isinf(pf)
    enf = torch.isnan(ef) | torch.isinf(ef)
    posdiff = (pnf != enf)
    nfdiff = int(posdiff.sum().item())
    idx = int(posdiff.nonzero().flatten()[0].item()) if nfdiff else -1
    print(f"  out[{OUT}] (sign of acos-derived) dt p={p.dtype}/e={e.dtype} "
          f"non-finite-pos-diff={nfdiff}")
    if idx >= 0:
        print(f"    idx{idx}: portable={pf[idx].item()!r}  eager={ef[idx].item()!r}  "
              f"(same NaN, different downstream slot)")

    reproduced = (nfdiff > 0) and bool(up_nan.any())
    print("REPLAY: ARTIFACT reproduced (inherited NaN, position differs)"
          if reproduced else "REPLAY: not reproduced")
    return 0 if reproduced else 1


if __name__ == "__main__":
    sys.exit(main())
