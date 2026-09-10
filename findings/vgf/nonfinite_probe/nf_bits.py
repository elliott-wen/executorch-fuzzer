#!/usr/bin/env python3
"""nf_bits.py — for selected NONFINITE rows, dump the RAW device bit patterns of the
diverging output, to test whether an "all non-finite" output is uninitialised/garbage
memory (a constant bit pattern) rather than a computed number-format effect.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nf_common import *

WANT = set(sys.argv[1:]) if len(sys.argv) > 1 else None
recs = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nf_all.jsonl"))]
for r in recs:
    key = r["job"]
    if WANT and key not in WANT:
        continue
    j = build_job(r["_srcB"], "vgf", False)
    if j.status != "READY":
        print(key, "BUILD", j.status); continue
    raws, err = vgf_run(j, "bits")
    if raws is None:
        print(key, "RUN", err); continue
    al = align(raws, j)
    p = r["_pB"] if r["_pB"] >= 0 else len(j.eager) - 1
    ref = j.eager[p]
    dev = al[p]
    b = dev.contiguous().view(torch.int32) if dev.dtype == torch.float32 else None
    hexs = ",".join(f"{x & 0xffffffff:08x}" for x in b.flatten()[:12].tolist()) if b is not None else "n/a"
    print(f"{key:12s} {r['tgt_op']:16s} dtype={str(dev.dtype).replace('torch.',''):8s} "
          f"ref={[round(float(x),4) for x in ref.flatten().float()[:6].tolist()]} "
          f"dev={[round(float(x),4) for x in dev.flatten().float()[:6].tolist()]}\n"
          f"{'':12s} dev_bits={hexs}  distinct_bits={len(set(b.flatten().tolist())) if b is not None else '?'}"
          f"  all_out_sizes={[len(x) for x in raws]} user_pos={j.user_pos} n_eager={len(j.eager)}", flush=True)
