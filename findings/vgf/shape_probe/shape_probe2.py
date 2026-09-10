#!/usr/bin/env python3
"""Probe the RECONSTRUCTED A/B graphs that produced the SHAPE verdicts.
For each case, build cone_localizer's build_src([]) (A side) and build_src(needed) (B side),
lower each, and compare eager output shape vs the .pte plan's DECLARED output shape.
PLAN!=EAGER  -> AOT shape disagreement on the reconstructed graph
PLAN==EAGER  -> a device shape diff would be runtime; also prints n_eager/user_pos for
                the select()/trim confound (device returning more outputs than eager)."""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from mobile.net import corpus as C
from mobile.gen.export import build_job
from mobile.gen.diff.cone_predicate import cone_localizer
from executorch.exir._serialize._program import deserialize_pte_binary

CO = "/data/jwen929/mobile/corpus_v4/vgf"
CASES = [
    ("w103:715", 3, "unfold_copy", ["n0","n4"]),
    ("w104:305", 0, "neg",         ["n0"]),
    ("w10:591",  1, "div",         ["n1","n4"]),
    ("w106:621", 0, "clamp",       ["n2"]),
    ("w107:200", 0, "cumsum",      ["n4","n3","n2","n1","n0"]),
    ("w107:319", 1, "log",         ["n1","n7","n3"]),
    ("w111:159", 1, "scatter_add", ["n0","n2"]),
    ("w114:256", 0, "min",         ["n3","n0","n2"]),
    ("w11:655",  0, "div",         ["n2"]),
    ("w120:300", 2, "squeeze_copy",["n0","n1","n5"]),
    ("w121:253", 1, "reciprocal",  ["n2"]),
    ("w22:81",   1, "view_copy",   ["n0","n3"]),
]

def plan_shapes(pte):
    prog = deserialize_pte_binary(pte)
    prog = getattr(prog, "program", prog)
    plan = prog.execution_plan[0]
    return [tuple(plan.values[i].val.sizes) if type(plan.values[i].val).__name__=="Tensor" else None
            for i in plan.outputs]

def one(job_id, tgt, src, label):
    try:
        j = build_job(src, "vgf", False)
    except Exception as e:
        return f"{label}: BUILD_EXC {type(e).__name__}"
    if j.status != "READY":
        return f"{label}: BUILD:{j.status} {j.detail[:40]}"
    eager = [tuple(t.shape) for t in j.eager]
    try: ps = plan_shapes(j.pte)
    except Exception as e: return f"{label}: PLAN_EXC {type(e).__name__}"
    up = j.user_pos
    sel = [ps[i] for i in up] if up and all(0 <= i < len(ps) for i in up) else ps[-len(eager):]
    ok = (sel == eager)
    return (f"{label}: deleg={j.delegated_ops} n_eager={len(eager)} n_plan_out={len(ps)} "
            f"user_pos={up} eager={eager} plan_all={ps} "
            f"{'PLAN==EAGER' if ok else '*** PLAN!=EAGER ***'}")

for job_id, out_idx, tgt, needed in CASES:
    print(f"\n=== {job_id} out{out_idx} target={tgt} needed_live={needed}")
    try:
        anc, build_src, target = cone_localizer(C.job_file(CO, job_id, "py"), out_idx)
    except Exception as e:
        print(f"  cone_localizer FAIL {type(e).__name__}: {e}"); continue
    print("  " + one(job_id, tgt, build_src([]), "A(baked)"))
    print("  " + one(job_id, tgt, build_src(needed), "B(live)"))
