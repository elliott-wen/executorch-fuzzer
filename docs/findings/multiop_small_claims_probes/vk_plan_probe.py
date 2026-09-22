#!/usr/bin/env python3
"""Host-only Vulkan AOT probe for Claim A.

For Bug(lift_fresh_copy) / BugClone(clone) / Control(add 0.0):
  1. lower to Vulkan (no GPU needed)
  2. intercept the DELEGATE-INTERNAL program right after the Vulkan
     MemoryPlanningPass and dump every TensorSpec's (mem_id, mem_offset,
     allocated_memory, lifetime) so we can see whether the sibling output
     and the still-live source actually OVERLAP in storage.
  3. optionally force the `verify_storage_reuse` gate open by handing
     MemoryPlanningPass an algo whose _callable_name() == "greedy".
"""
import os, sys, warnings, logging, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
import executorch.backends.vulkan.vulkan_preprocess as VP
from executorch.exir.passes.memory_planning_pass import _callable_name
from executorch.exir.memory_planning import MemoryPlanningAlgorithmSuite, greedy


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


CAPTURED = []

def install_dump():
    orig = VP.VkGraphBuilder
    class Dumping(orig):
        def __init__(self, program, *a, **kw):
            gm = program.graph_module
            rows = []
            for n in gm.graph.nodes:
                spec = n.meta.get("spec", None)
                specs = spec if isinstance(spec, (list, tuple)) else ([spec] if spec is not None else [])
                for i, s in enumerate(specs):
                    if s is None or not hasattr(s, "mem_offset"):
                        continue
                    rows.append(dict(node=n.name, op=str(n.op), tgt=str(n.target),
                                     idx=i,
                                     mem_id=s.mem_id, mem_offset=s.mem_offset,
                                     nbytes=getattr(s, "allocated_memory", None),
                                     lifetime=tuple(s.lifetime) if s.lifetime else None,
                                     shape=tuple(s.shape) if s.shape is not None else None,
                                     dtype=str(s.dtype)))
            CAPTURED.append(dict(graph=[ (n.name, n.op, str(n.target)) for n in gm.graph.nodes ],
                                 rows=rows,
                                 outputs=[a.name if hasattr(a,'name') else str(a)
                                          for a in (list(gm.graph.output_node().args[0])
                                                    if gm.graph.output_node().args else [])]))
            super().__init__(program, *a, **kw)
    VP.VkGraphBuilder = Dumping


def open_gate():
    """Replace MemoryPlanningAlgorithmSuite with a factory whose product is a
    plain function literally named `greedy`, so _callable_name() == 'greedy'
    and the unconditional verify_storage_reuse() gate in
    exir/passes/memory_planning_pass.py opens."""
    RealSuite = MemoryPlanningAlgorithmSuite
    def factory(algo_list=None):
        suite = RealSuite(algo_list=algo_list)
        def greedy(alignment, specs, graph_module, graph_signature, extra_padding):
            return suite(alignment, specs, graph_module, graph_signature, extra_padding)
        return greedy
    VP.MemoryPlanningAlgorithmSuite = factory


def lower(M, x):
    ep = torch.export.export(M(), (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False)).to_executorch()


def show(name, cap):
    print(f"--- {name}: delegate-internal graph ({len(cap['graph'])} nodes)")
    for nm, op, tgt in cap["graph"]:
        print(f"      {op:14s} {nm:28s} {tgt}")
    print(f"    outputs: {cap['outputs']}")
    print(f"    {'node':28s} {'mem_id':>6s} {'offset':>8s} {'bytes':>7s} {'lifetime':>12s}  shape/dtype")
    for r in cap["rows"]:
        print(f"    {r['node']:28s} {str(r['mem_id']):>6s} {str(r['mem_offset']):>8s} "
              f"{str(r['nbytes']):>7s} {str(r['lifetime']):>12s}  {r['shape']} {r['dtype']}")
    # overlap report
    rows = [r for r in cap["rows"] if r["mem_offset"] is not None and r["nbytes"]]
    print("    STORAGE OVERLAP PAIRS (same mem_id, byte ranges intersect):")
    found = False
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if a["mem_id"] != b["mem_id"]: continue
            a0, a1 = a["mem_offset"], a["mem_offset"] + a["nbytes"] - 1
            b0, b1 = b["mem_offset"], b["mem_offset"] + b["nbytes"] - 1
            if a0 <= b1 and b0 <= a1:
                lt_ov = (a["lifetime"] and b["lifetime"] and
                         a["lifetime"][0] <= b["lifetime"][1] and b["lifetime"][0] <= a["lifetime"][1])
                found = True
                print(f"      {a['node']} [{a0}-{a1}] lt={a['lifetime']}  X  "
                      f"{b['node']} [{b0}-{b1}] lt={b['lifetime']}"
                      f"   {'*** LIFETIME ALSO OVERLAPS ***' if lt_ov else '(lifetimes disjoint: legal)'}")
    if not found:
        print("      none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="force verify_storage_reuse gate open")
    args = ap.parse_args()

    print("_callable_name(MemoryPlanningAlgorithmSuite()) =",
          repr(_callable_name(MemoryPlanningAlgorithmSuite()))[:90])
    from functools import partial
    print("_callable_name(partial(greedy, allow_overlapping_allocations=False)) =",
          repr(_callable_name(partial(greedy, allow_overlapping_allocations=False))))
    print("gate opens for suite instance?",
          _callable_name(MemoryPlanningAlgorithmSuite()) == "greedy")
    print()

    install_dump()
    if args.gate:
        open_gate()
        print(">>> GATE FORCED OPEN (algo _callable_name == 'greedy')\n")

    x = torch.tensor([-0.4824913740158081, 1.1103534698486328])
    for name, M in [("Bug(lift_fresh_copy)", Bug), ("BugClone(clone)", BugClone),
                    ("Control(add 0.0)", Control)]:
        CAPTURED.clear()
        print(f"=== {name}")
        try:
            lower(M, x)
        except Exception as e:
            print(f"    LOWERING RAISED: {type(e).__name__}: {e}")
            for c in CAPTURED: show(name, c)
            print()
            continue
        for c in CAPTURED:
            show(name, c)
        print()

main()
