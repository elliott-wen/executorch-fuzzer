#!/usr/bin/env python3
"""Test the POWER of the SCALE consistency test for the 3 Ethos-U requant-scale cases.
deep_graphopt.py accepts SCALE when max(ratios)-min(ratios) < 0.05*|mean|, over
  ratios = [dev[k]/ref[k] for k where |ref[k]|>1e-3]
With a single surviving ratio the test is vacuous (max-min == 0). The surviving count
depends only on the REFERENCE, which is host-computable -- no FVP needed."""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.executor import corpus as C
from mobile.gen.export import build_job
from mobile.gen.diff.cone_predicate import output_set_localizer

CO = "/data/jwen929/mobile/corpus_v2/ethos-u"
# job, out_idx, breaking sibling, reported factor, target op
CASES = [
    ("w14:133", 3, "n3", -3.793,      "gelu.out"),
    ("w23:956", 3, "n5",  0.1081,     "logit.default"),
    ("w36:1",   2, "n7", -1.537e18,   "logit.default"),
]

for job_id, out_idx, sib, factor, op in CASES:
    print(f"\n=== {job_id} out{out_idx} target_op={op} sibling={sib} reported=SCALE:{factor:g}")
    try:
        others, bos, target, ret = output_set_localizer(C.job_file(CO, job_id, "py"), out_idx)
    except Exception as e:
        print(f"  localizer FAIL {type(e).__name__}: {e}"); continue
    print(f"  target={target} ret={ret}")
    if sib not in ret:
        print(f"  !! sibling {sib} not in ret"); continue
    try:
        j = build_job(bos([target, sib]), "ethos-u", True)
    except Exception as e:
        print(f"  BUILD_EXC {type(e).__name__}: {str(e)[:80]}"); continue
    if j.status != "READY":
        print(f"  BUILD:{j.status} {j.detail[:70]}"); continue
    sel = [o for o in ret if o in {target, sib}]
    pos = sel.index(target)
    ref = j.eager[pos]
    rt = [float(x) for x in ref.flatten().tolist()]
    kept = [x for x in rt if abs(x) > 1e-3]
    print(f"  ref shape={tuple(ref.shape)} dtype={ref.dtype} numel={len(rt)}")
    print(f"  ref values[:8]={[round(x,6) for x in rt[:8]]}")
    print(f"  ratios surviving |ref|>1e-3 : {len(kept)} of {len(rt)}"
          f"   -> SCALE test {'VACUOUS (single ratio, max-min==0)' if len(kept)<=1 else 'has power'}")
    if kept:
        implied = [factor * x for x in kept[:4]]
        print(f"  implied device values = factor*ref = {[f'{v:.4g}' for v in implied]}")
