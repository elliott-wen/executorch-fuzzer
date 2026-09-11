#!/usr/bin/env python3
"""Replay: max_pool2d_with_indices_backward OOB index -> portable SEGFAULT (exit 139).

This is NOT a born-here value bug: the portable scatter is a line-for-line clone of
ATen's CPU backward (`grad_input[maxindex] += grad_output`, no bounds check). The real
hardening defect is the UNGUARDED, out-of-bounds write: a wild index (1,000,000 into a
9-element plane) makes the portable runtime do an OOB write and SEGFAULT, while eager
silently drops the out-of-range contribution and runs fine. (The value mismatches in the
md are argument artifacts; this replay demonstrates the OOB-write crash.)

Strategy: confirm eager runs OK in-process (drops the OOB write), then export+run the
portable side IN A CHILD subprocess and assert it died with a fatal signal (rc<0, SIGSEGV).

Usage: cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_maxpool2d-backward.py
(filter noise: | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference")
"""
import os
import subprocess
import sys
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")  # so `import mobile` resolves
warnings.filterwarnings("ignore")

import torch

# input plane is 3x3 = 9 elements; output 2x2; one index is wildly OOB (1,000,000).
KERNEL = [2, 2]
STRIDE = [1, 1]
PADDING = [0, 0]
DILATION = [1, 1]
CEIL = False


def make_args():
    inp = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    grad_output = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    # last index is OOB for a 9-element plane
    indices = torch.tensor([[[[0, 1], [3, 1000000]]]], dtype=torch.int64)
    return inp, grad_output, indices


def eager_check():
    inp, go, idx = make_args()
    out = torch.ops.aten.max_pool2d_with_indices_backward.default(
        go, inp, KERNEL, STRIDE, PADDING, DILATION, CEIL, idx)
    print("=== eager (in-process) ===")
    print("  grad_input:", out.flatten().tolist())
    print("  RAN OK (OOB contribution dropped, no crash)")
    return True


CHILD = r"""
import os, sys, warnings
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
warnings.filterwarnings("ignore")
import torch
from torch.export import export
from executorch.exir import to_edge
from mobile.executor import et_runner

KERNEL=[2,2]; STRIDE=[1,1]; PADDING=[0,0]; DILATION=[1,1]; CEIL=False

class M(torch.nn.Module):
    def forward(self, go, inp, idx):
        return torch.ops.aten.max_pool2d_with_indices_backward.default(
            go, inp, KERNEL, STRIDE, PADDING, DILATION, CEIL, idx)

inp = torch.arange(9, dtype=torch.float32).reshape(1,1,3,3)
go = torch.tensor([[[[1.0,2.0],[3.0,4.0]]]], dtype=torch.float32)
idx = torch.tensor([[[[0,1],[3,1000000]]]], dtype=torch.int64)
exe = to_edge(export(M().eval(), (go, inp, idx))).to_executorch()
out = et_runner.run_pte(exe.buffer, [go.clone(), inp.clone(), idx.clone()])  # OOB write here
print("RAN OK (no crash):", out[0].flatten().tolist())
"""


def portable_child():
    print("=== portable (child subprocess) ===")
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True)
    rc = r.returncode
    sig = -rc if rc < 0 else None
    code = 128 + sig if sig is not None else rc
    tail = (r.stderr.strip().splitlines() or [""])[-1]
    print(f"  child exit={code}  (SIGSEGV=139 expected)")
    print(f"  last stderr: {tail}")
    return rc < 0  # died on a fatal signal


def main():
    e_ok = eager_check()
    print()
    crashed = portable_child()
    print()
    reproduced = e_ok and crashed
    print("REPLAY: OOB-write SEGFAULT REPRODUCED" if reproduced
          else "REPLAY: did NOT reproduce")
    return 0 if reproduced else 1


if __name__ == "__main__":
    sys.exit(main())
