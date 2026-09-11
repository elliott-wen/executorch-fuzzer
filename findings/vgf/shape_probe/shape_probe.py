#!/usr/bin/env python3
"""For each VGF SHAPE case: build the job, and compare
  (a) eager output shapes  vs  (b) the lowered .pte plan's DECLARED output shapes at user_pos.
If (a)!=(b) -> AOT shape-inference disagreement (plan is already wrong).
If (a)==(b) -> plan agrees with eager; a device shape diff would be runtime/IO-metadata or misalignment.
Also reports user_pos and the full plan output list so misalignment is visible."""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from mobile.executor import corpus as C
from mobile.gen.export import build_job

CO = "/data/jwen929/mobile/corpus_v4/vgf"
CASES = [
    ("w103:715", 3, "unfold_copy"), ("w104:305", 0, "neg"),
    ("w10:591", 1, "div"),          ("w106:621", 0, "clamp"),
    ("w107:200", 0, "cumsum"),      ("w107:319", 1, "log"),
    ("w111:159", 1, "scatter_add"), ("w114:256", 0, "min"),
    ("w11:655", 0, "div"),          ("w120:300", 2, "squeeze_copy"),
    ("w121:253", 1, "reciprocal"),  ("w22:81", 1, "view_copy"),
]

def plan_out_shapes(pte):
    from executorch.exir._serialize._program import deserialize_pte_binary
    prog = deserialize_pte_binary(pte)
    prog = getattr(prog, "program", prog)
    plan = prog.execution_plan[0]
    out = []
    for vi in plan.outputs:
        v = plan.values[vi].val
        out.append(tuple(v.sizes) if type(v).__name__ == "Tensor" else None)
    return out

for job_id, out_idx, tgt in CASES:
    try:
        src = open(C.job_file(CO, job_id, "py")).read()
    except Exception as e:
        print(f"{job_id:10s} out{out_idx:<2} {tgt:14s} SRC_FAIL {e}"); continue
    try:
        j = build_job(src, "vgf", False)
    except Exception as e:
        print(f"{job_id:10s} out{out_idx:<2} {tgt:14s} BUILD_EXC {type(e).__name__}: {str(e)[:70]}"); continue
    if j.status != "READY":
        print(f"{job_id:10s} out{out_idx:<2} {tgt:14s} BUILD:{j.status} {j.detail[:60]}"); continue
    eager = [tuple(t.shape) for t in j.eager]
    try:
        pshapes = plan_out_shapes(j.pte)
    except Exception as e:
        print(f"{job_id:10s} out{out_idx:<2} {tgt:14s} PLAN_EXC {type(e).__name__}: {str(e)[:60]}"); continue
    up = j.user_pos
    sel = [pshapes[i] for i in up] if up and all(0 <= i < len(pshapes) for i in up) else pshapes[-len(eager):]
    agree = (sel == eager)
    print(f"{job_id:10s} out{out_idx:<2} {tgt:14s} deleg={j.delegated_ops:<3} "
          f"user_pos={up} n_plan_out={len(pshapes)} n_eager={len(eager)} "
          f"{'PLAN==EAGER' if agree else 'PLAN!=EAGER'}")
    if not agree:
        for k, (e, s) in enumerate(zip(eager, sel)):
            flag = "  <<<" if e != s else ""
            print(f"              out[{k}] eager={e} plan={s}{flag}")
