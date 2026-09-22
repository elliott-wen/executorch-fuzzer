#!/usr/bin/env python3
"""Claim C/D empirical probe: run aliasing-shaped multi-output graphs
IN-PROCESS on one backend and look for dropped stores / wrong siblings.
usage: alias_probe.py <backend>
"""
import os, sys, warnings, logging
BK = sys.argv[1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ["MOBILE_BACKENDS"] = BK
sys.path.insert(0,"/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.gen.export import build_job
from mobile.executor.et_runner import run_pte
from mobile.executor import compare as cmp

HDR = '''import torch
torch.manual_seed(12648430)
def _first(r):
    if isinstance(r, torch.Tensor): return r
    if isinstance(r, (tuple, list)):
        for x in r:
            if isinstance(x, torch.Tensor): return x
    return r
L0 = torch.randn((4, 8))
def g(L0):
'''
FTR = "\nLEAVES = [L0]\n"

CASES = {
 # name: (body lines, return expr)
 "clone_sibling": (["n0 = torch.ops.aten.sigmoid.default(L0)",
                    "n1 = torch.ops.aten.clone.default(n0)",
                    "n2 = torch.ops.aten.mul.Tensor(n0, 2.0)"], "n1, n2"),
 "clone_plus_reduce": (["n0 = torch.ops.aten.sigmoid.default(L0)",
                    "n1 = torch.ops.aten.clone.default(n0)",
                    "n2 = torch.ops.aten.acos.default(n0)",
                    "n3 = torch.ops.aten.prod.default(n2)"], "n3, n1"),
 "alias_copy_sibling": (["n0 = torch.ops.aten.sigmoid.default(L0)",
                    "n1 = torch.ops.aten.alias_copy.default(n0)",
                    "n2 = torch.ops.aten.mul.Tensor(n0, 2.0)"], "n1, n2"),
 "expand_copy_sibling": (["n0 = torch.ops.aten.sigmoid.default(L0)",
                    "n1 = torch.ops.aten.expand_copy.default(n0, [4, 8])",
                    "n2 = torch.ops.aten.mul.Tensor(n0, 2.0)"], "n1, n2"),
 "view_copy_sibling": (["n0 = torch.ops.aten.sigmoid.default(L0)",
                    "n1 = torch.ops.aten.view_copy.default(n0, [8, 4])",
                    "n2 = torch.ops.aten.mul.Tensor(n0, 2.0)"], "n1, n2"),
 "same_node_twice": (["n0 = torch.ops.aten.sigmoid.default(L0)",
                    "n1 = torch.ops.aten.mul.Tensor(n0, 2.0)"], "n0, n1, n0"),
 "input_co_returned": (["n0 = torch.ops.aten.sigmoid.default(L0)"], "L0, n0"),
 "clone_of_input": (["n0 = torch.ops.aten.clone.default(L0)",
                    "n1 = torch.ops.aten.sigmoid.default(L0)"], "n0, n1"),
 "three_aliases": (["n0 = torch.ops.aten.sigmoid.default(L0)",
                    "n1 = torch.ops.aten.clone.default(n0)",
                    "n2 = torch.ops.aten.alias_copy.default(n0)",
                    "n3 = torch.ops.aten.acos.default(n0)",
                    "n4 = torch.ops.aten.prod.default(n3)"], "n4, n1, n2, n0"),
 "lift_fresh_repro": (["n0 = torch.ops.aten.sigmoid.default(L0)",
                    "n1 = torch.ops.aten.clone.default(n0)",
                    "n2 = torch.ops.aten.acos.default(n0)",
                    "n3 = torch.ops.aten.prod.default(n2)",
                    "n4 = torch.ops.aten.acosh.default(n1)"], "n3, n4"),
}
print(f"### backend={BK}")
for name,(body,ret) in CASES.items():
    src = HDR + "\n".join("    "+f"{l.split('=')[0].strip()} = _first({l.split('=',1)[1].strip()})" for l in body) \
        + f"\n    return ({ret},)\n" + FTR
    try:
        j = build_job(src, BK, False)
    except Exception as e:
        print(f"{name:22s} BUILD_EXC {type(e).__name__}: {str(e)[:80]}"); continue
    if j.status != "READY":
        print(f"{name:22s} {j.status} {j.detail[:80]}"); continue
    try:
        out = run_pte(j.pte, j.inputs)
    except Exception as e:
        print(f"{name:22s} RUN_EXC {type(e).__name__}: {str(e)[:80]}"); continue
    et = list(out)
    if len(et) > len(j.eager): et = et[len(et)-len(j.eager):]
    verd = []
    for i,(e,d) in enumerate(zip(j.eager, et)):
        s,det = cmp._cmp(e,d,i)
        zero = bool(torch.as_tensor(d).float().abs().max()==0) and bool(torch.as_tensor(e).float().abs().max()>0)
        verd.append(f"out{i}:{s}{'/ZEROED' if zero else ''}{(' '+det) if s!='OK' else ''}")
    print(f"{name:22s} deleg={j.delegated_ops} n_et={len(out)} n_eager={len(j.eager)} | " + " | ".join(verd))
