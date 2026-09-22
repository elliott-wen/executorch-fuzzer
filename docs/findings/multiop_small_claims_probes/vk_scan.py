#!/usr/bin/env python3
"""Corpus-wide host-only measurement of the Vulkan delegate output-arity defect:
for each delegate blob, compare len(VkGraph.output_ids) (what the runtime will
write) against the number of output slots the DelegateCall passes (args after
the declared inputs). A shortfall => the leading output slot(s) are never
written by VulkanBackend.cpp (output_offset = args.size() - num_outputs)."""
import os, sys, warnings, logging, random, json, traceback
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MOBILE_BACKENDS", "vulkan")
sys.path.insert(0, "/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from pathlib import Path
import executorch.backends.vulkan.vulkan_preprocess as VP
from mobile.gen.export import build_job

REC = []
_orig = VP.serialize_vulkan_graph
def traced(vk_graph, *a, **kw):
    REC.append((len(vk_graph.input_ids), len(vk_graph.output_ids),
                len(set(vk_graph.output_ids))))
    return _orig(vk_graph, *a, **kw)
VP.serialize_vulkan_graph = traced

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
CO = sys.argv[2] if len(sys.argv) > 2 else "/data/jwen929/mobile/corpus_v2/vulkan"
files = sorted(Path(CO).glob("w*/*.py"))
random.seed(1234); random.shuffle(files)

from executorch.exir._serialize._program import deserialize_pte_binary

stats = dict(tried=0, ready=0, notready=0, exc=0, deleg=0, mismatch_blobs=0,
             mismatch_jobs=0, dup_outid_blobs=0)
examples = []
for f in files:
    if stats["tried"] >= N: break
    stats["tried"] += 1
    REC.clear()
    try:
        j = build_job(f.read_text(), "vulkan", False)
    except Exception:
        stats["exc"] += 1; continue
    if j.status != "READY":
        stats["notready"] += 1; continue
    stats["ready"] += 1
    try:
        prog = deserialize_pte_binary(j.pte); prog = getattr(prog, "program", prog)
        plan = prog.execution_plan[0]
        calls = [i.instr_args for i in plan.chains[0].instructions
                 if type(i.instr_args).__name__ == "DelegateCall"]
    except Exception:
        stats["exc"] += 1; continue
    if len(calls) != len(REC):
        continue  # can't align reliably
    bad = False
    for (nin, nout, nuniq), c in zip(REC, calls):
        stats["deleg"] += 1
        slots = len(c.args) - nin
        if nout != nuniq:
            stats["dup_outid_blobs"] += 1
        if slots != nout:
            stats["mismatch_blobs"] += 1; bad = True
            if len(examples) < 25:
                examples.append((f.name, nin, nout, len(c.args), slots))
    if bad:
        stats["mismatch_jobs"] += 1
    if stats["tried"] % 25 == 0:
        print(json.dumps(stats), flush=True)

print("FINAL", json.dumps(stats))
for e in examples: print("EX", e)
