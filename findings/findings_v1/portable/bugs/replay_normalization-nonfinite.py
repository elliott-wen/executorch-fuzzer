#!/usr/bin/env python3
"""Replay: _native_batch_norm_legit_no_training var=0,eps=0 — portable invstd=+inf
=> (x-mean)*inf = +-inf, eager guards var==0&&eps==0 => invstd=0 => nan.

Portable computes `invstd = 1/sqrt(var+eps)` with no guard; ATen's InvStd forces
invstd=0 when var==0 && eps==0. With finite x where (x-mean)!=0, var=0, eps=0:
eager -> nan, portable -> +-inf (finiteness differs).

Usage: cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_normalization-nonfinite.py
(filter noise: | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference")
"""
import math
import os
import sys
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")  # so `import mobile` resolves
warnings.filterwarnings("ignore")

import torch
from torch.export import export
from executorch.exir import to_edge
from mobile.net import et_runner


class M(torch.nn.Module):
    def forward(self, x, mean, var):
        # weight=None, bias=None, momentum=0.0, eps=0.0
        out = torch.ops.aten._native_batch_norm_legit_no_training.default(
            x, None, None, mean, var, 0.0, 0.0)
        return out[0]


def finiteness_diff(p, e):
    pf, ef = p.flatten(), e.flatten()
    out = []
    for i in range(pf.numel()):
        pv, ev = pf[i].item(), ef[i].item()
        if math.isnan(pv) != math.isnan(ev) or math.isinf(pv) != math.isinf(ev):
            out.append((i, pv, ev))
    return out


def main():
    # x shape (N=1, C=1): (x-mean)=2 != 0; var=0; eps=0 -> eager nan, portable +inf.
    x = torch.tensor([[2.0]], dtype=torch.float32)
    mean = torch.tensor([0.0], dtype=torch.float32)
    var = torch.tensor([0.0], dtype=torch.float32)
    exe = to_edge(export(M().eval(), (x, mean, var))).to_executorch()
    p = et_runner.run_pte(exe.buffer, [x.clone(), mean.clone(), var.clone()])[0]
    e = M()(x, mean, var)
    print("x:", x.tolist(), "mean:", mean.tolist(), "var:", var.tolist(), "eps=0")
    print("eager:   ", e.flatten().tolist())
    print("portable:", p.flatten().tolist())
    diffs = finiteness_diff(p, e)
    reproduced = False
    for i, pv, ev in diffs:
        print(f"  [{i}] portable={pv!r}  eager={ev!r}")
        # documented: portable +-inf where eager nan
        if math.isinf(pv) and math.isnan(ev):
            reproduced = True
    print("REPLAY: bug REPRODUCED" if reproduced else "REPLAY: no divergence")
    return 0 if reproduced else 1


if __name__ == "__main__":
    sys.exit(main())
