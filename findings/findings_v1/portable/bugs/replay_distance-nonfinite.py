#!/usr/bin/env python3
"""Replay: cdist p=0 (L0) NaN handling — portable counts NaN as 1 (finite),
eager propagates NaN.

Portable L0 map is `diff==0 ? 0 : 1`, so a NaN diff (nan==0 is false) becomes a
finite count; ATen uses `min(ceil(abs(diff)),1)` which yields NaN. We export a
minimal 2x2-vs-2x2 cdist(p=0) where one x1 row carries a NaN, so the NaN-involving
pairs are nan in eager but a finite count in portable.

Usage: cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_distance-nonfinite.py
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
from mobile.executor import et_runner


class M(torch.nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=0.0)


def finiteness_diff(p, e):
    """positions nan-in-one / finite-in-other (or inf-vs-nan)."""
    pf, ef = p.flatten(), e.flatten()
    out = []
    for i in range(pf.numel()):
        pv, ev = pf[i].item(), ef[i].item()
        pn, en = math.isnan(pv), math.isnan(ev)
        if pn != en:  # one is nan, the other is not
            out.append((i, pv, ev))
    return out


def main():
    # row 0 of x1 carries a NaN -> the pair against any x2 row produces a NaN diff.
    x1 = torch.tensor([[float("nan"), 0.0],
                       [1.0, 1.0]], dtype=torch.float32)
    x2 = torch.tensor([[0.0, 0.0],
                       [2.0, 2.0]], dtype=torch.float32)
    exe = to_edge(export(M().eval(), (x1, x2))).to_executorch()
    p = et_runner.run_pte(exe.buffer, [x1.clone(), x2.clone()])[0]
    e = M()(x1, x2)
    print("x1:", x1.tolist())
    print("x2:", x2.tolist())
    print("eager:   ", e.flatten().tolist())
    print("portable:", p.flatten().tolist())
    diffs = finiteness_diff(p, e)
    reproduced = False
    for i, pv, ev in diffs:
        print(f"  [{i}] portable={pv!r}  eager={ev!r}")
        # documented: eager nan, portable finite
        if math.isnan(ev) and math.isfinite(pv):
            reproduced = True
    print("REPLAY: bug REPRODUCED" if reproduced else "REPLAY: no divergence")
    return 0 if reproduced else 1


if __name__ == "__main__":
    sys.exit(main())
