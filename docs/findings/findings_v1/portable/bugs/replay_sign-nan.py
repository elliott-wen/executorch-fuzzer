#!/usr/bin/env python3
"""Replay: aten::sign(NaN) — portable propagates NaN, eager returns 0.

Two self-contained demonstrations:
  (A) Corpus job w0:1490 — non-finite POSITION diff at out[0]: portable=nan
      where eager is finite (sign(NaN) launders into atan2 -> nan).
  (B) Isolated export proof — torch.sign on [nan,inf,-inf,0,-0,-3,2]:
      portable sign(nan)=nan vs eager 0.0 (born-here root cause).

Usage: cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_sign-nan.py
(filter noise: | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference")
"""
import importlib.util as u
import os
import sys
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")  # so `import mobile` resolves
warnings.filterwarnings("ignore")

import torch
from torch.export import export
from executorch.exir import to_edge
from mobile.executor import corpus, protocol, et_runner

REPO = "/data/jwen929/mobile"
JOB = "corpus/portable/w0/w0_1490.py"   # logit -> sign(NaN) -> atan2
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
    outs = m.g(*[t.clone() for t in inputs])
    return [x for x in outs if hasattr(x, "dtype")]


def finiteness_diff(p, e):
    """Positions where p/e differ in NaN/inf/finite classification."""
    pf, ef = p.flatten(), e.flatten()
    diffs = []
    for i in range(pf.numel()):
        pv, ev = pf[i].item(), ef[i].item()
        # classify: 0=finite, 1=nan, 2=+inf, 3=-inf
        def cls(v):
            import math
            if math.isnan(v):
                return 1
            if math.isinf(v):
                return 2 if v > 0 else 3
            return 0
        if cls(pv) != cls(ev):
            diffs.append((i, pv, ev))
    return diffs


def corpus_replay():
    print("=== (A) Corpus job w0:1490 — non-finite position diff ===")
    job_id, pte, inputs = load(JOB)
    pouts = et_runner.run_pte(pte, inputs)
    eouts = eager(JOB, inputs)
    p, e = pouts[OUT], eouts[OUT]
    diffs = finiteness_diff(p, e)
    print(f"job {job_id} out[{OUT}] dt p={p.dtype}/e={e.dtype} "
          f"nonfinite-pos-diff={len(diffs)}")
    import math
    reproduced = False
    for i, pv, ev in diffs:
        print(f"  [{i}] portable={pv!r}  eager={ev!r}")
        if math.isnan(pv) and math.isfinite(ev):
            reproduced = True
    print("  CORPUS: REPRODUCED" if reproduced else "  CORPUS: no finiteness divergence")
    return reproduced


def isolated_proof():
    print("=== (B) Isolated export proof — torch.sign special values ===")

    class M(torch.nn.Module):
        def forward(self, x):
            return torch.sign(x)

    x = torch.tensor([float("nan"), float("inf"), -float("inf"),
                      0.0, -0.0, -3.0, 2.0], dtype=torch.float32)
    exe = to_edge(export(M().eval(), (x,))).to_executorch()
    p = et_runner.run_pte(exe.buffer, [x.clone()])[0]
    e = M()(x)
    print("input   :", x.tolist())
    print("eager   :", e.tolist())
    print("portable:", p.tolist())
    import math
    reproduced = math.isnan(p[0].item()) and e[0].item() == 0.0
    print(f"  pos0: portable sign(nan)={p[0].item()!r}  eager={e[0].item()!r}")
    print("  ISOLATED: REPRODUCED" if reproduced else "  ISOLATED: no divergence")
    return reproduced


def main():
    a = corpus_replay()
    print()
    b = isolated_proof()
    print()
    ok = a or b
    print("REPLAY: bug REPRODUCED" if ok else "REPLAY: no divergence")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
