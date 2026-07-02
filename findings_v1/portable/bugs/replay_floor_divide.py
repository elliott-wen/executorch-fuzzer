#!/usr/bin/env python3
"""Replay: aten::floor_divide by-zero — portable 0.0//0.0 = +inf, eager = nan.

Minimal exported graph (no corpus job needed; the bug is born in the portable
math_util.h functor which returns signbit(a)?-inf:+inf at b==0 instead of a/b).

Usage: cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_floor_divide.py
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
    def forward(self, a, b):
        return torch.floor_divide(a, b)


def run(dtype):
    a = torch.tensor([0., 1., -1., 5.], dtype=dtype)
    b = torch.tensor([0., 0., 0., 2.], dtype=dtype)
    exe = to_edge(export(M().eval(), (a, b))).to_executorch()
    p = et_runner.run_pte(exe.buffer, [a.clone(), b.clone()])[0]
    e = M()(a, b)
    print(f"--- {dtype} a={a.tolist()} b={b.tolist()} ---")
    print("  portable:", p.tolist())
    print("  eager:   ", e.tolist())
    p0, e0 = p[0].item(), e[0].item()
    print(f"  pos0: portable={p0!r}  eager={e0!r}")
    # documented divergence: portable[0] is +inf, eager[0] is nan
    return math.isinf(p0) and p0 > 0 and math.isnan(e0)


def main():
    ok32 = run(torch.float32)
    print()
    ok16 = run(torch.float16)
    print()
    ok = ok32 and ok16
    print("REPLAY: bug REPRODUCED" if ok else "REPLAY: no divergence")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
