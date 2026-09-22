#!/usr/bin/env python3
"""Decisive Claim-A experiment: force the verify_storage_reuse gate open in the
Vulkan delegate's own MemoryPlanningPass and see whether it RAISES."""
import os, sys, warnings, logging, argparse, traceback
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
import executorch.backends.vulkan.vulkan_preprocess as VP
import executorch.exir.passes.memory_planning_pass as MPP
from executorch.exir.memory_planning import MemoryPlanningAlgorithmSuite, Verifier

CALLS = []
_orig_vsr = Verifier.verify_storage_reuse
def traced_vsr(self, allow=False):
    tag = f"verify_storage_reuse(allow={allow}) on {len(list(self.graph_module.graph.nodes))}-node gm"
    try:
        n = _orig_vsr(self, allow)
        CALLS.append(f"OK   {tag} -> {n} reuse pairs")
        return n
    except Exception as e:
        CALLS.append(f"RAISE {tag} -> {type(e).__name__}: {e}")
        raise
Verifier.verify_storage_reuse = traced_vsr

class Bug(torch.nn.Module):
    def forward(self, x):
        n0 = torch.ops.aten.sigmoid.default(x)
        n1 = torch.ops.aten.lift_fresh_copy.default(n0)
        out0 = torch.ops.aten.prod.default(torch.ops.aten.acos.default(n0).to(torch.float16))
        out1 = torch.ops.aten.acosh.default(n1).to(torch.float16)
        return out0, out1
class BugClone(torch.nn.Module):
    def forward(self, x):
        n0 = torch.ops.aten.sigmoid.default(x)
        n1 = torch.ops.aten.clone.default(n0)
        out0 = torch.ops.aten.prod.default(torch.ops.aten.acos.default(n0).to(torch.float16))
        out1 = torch.ops.aten.acosh.default(n1).to(torch.float16)
        return out0, out1
class Control(torch.nn.Module):
    def forward(self, x):
        n0 = torch.ops.aten.sigmoid.default(x)
        n1 = torch.ops.aten.add.Tensor(n0, 0.0)
        out0 = torch.ops.aten.prod.default(torch.ops.aten.acos.default(n0).to(torch.float16))
        out1 = torch.ops.aten.acosh.default(n1).to(torch.float16)
        return out0, out1
# a bigger, more realistic aliasing graph: elidable copy of a shared tensor,
# many live siblings so the planner has real reuse pressure
class Big(torch.nn.Module):
    def forward(self, x):
        n0 = torch.ops.aten.sigmoid.default(x)
        n1 = torch.ops.aten.clone.default(n0)             # elided
        n2 = torch.ops.aten.expand_copy.default(n0, [4, 8])
        a = torch.ops.aten.acos.default(n0)
        b = torch.ops.aten.mul.Tensor(a, n0)
        c = torch.ops.aten.add.Tensor(b, n1)
        d = torch.ops.aten.sub.Tensor(n2, 1.0)
        return torch.ops.aten.prod.default(c), n1, d, n0

def open_gate():
    RealSuite = MemoryPlanningAlgorithmSuite
    def factory(algo_list=None):
        suite = RealSuite(algo_list=algo_list)
        def greedy(alignment, specs, gm, gs, extra_padding):
            return suite(alignment, specs, gm, gs, extra_padding)
        return greedy
    VP.MemoryPlanningAlgorithmSuite = factory

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    if a.gate: open_gate(); print(">>> GATE FORCED OPEN")
    else: print(">>> gate as-shipped (suite instance)")
    cases = [("Bug", Bug, torch.tensor([-0.4824913740158081, 1.1103534698486328])),
             ("BugClone", BugClone, torch.tensor([-0.4824913740158081, 1.1103534698486328])),
             ("Control", Control, torch.tensor([-0.4824913740158081, 1.1103534698486328])),
             ("Big", Big, torch.randn(1, 8))]
    for name, M, x in cases:
        CALLS.clear()
        print(f"\n=== {name}")
        try:
            ep = torch.export.export(M(), (x,))
            to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()],
                compile_config=EdgeCompileConfig(_check_ir_validity=False)).to_executorch()
            print("   lowering: OK")
        except Exception as e:
            print(f"   lowering: {type(e).__name__}: {str(e)[:300]}")
        for c in CALLS: print("   " + c)
        if not CALLS: print("   (verify_storage_reuse was NEVER CALLED)")
main()
