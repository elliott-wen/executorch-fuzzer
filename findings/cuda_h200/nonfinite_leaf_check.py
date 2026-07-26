#!/usr/bin/env python3
"""nonfinite_leaf_check.py — step 3e: for each job_id, was the LEAF input already non-finite?

For a single-op graph, nan/inf on device where the reference is finite is the op itself — UNLESS
the input leaf was already non-finite (then a matching-position nan/inf is not a device bug). This
loads each job's own LEAVES from its corpus .py and reports leaf finiteness, so non-finite MISMATCH
findings can be split into 'op emits nan/inf from FINITE input' (real) vs 'leaf already non-finite'.

Usage: nonfinite_leaf_check.py <job_id> [<job_id> ...]   (or pipe job_ids on stdin, one per line)
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/data/jwen929")
import torch
from pathlib import Path
from mobile.gen.export.job import load_functional_source

CORPUS = Path("/data/jwen929/mobile/corpus_v3/cuda")

def py_for(job_id):
    prod, idx = job_id.split(":")
    return CORPUS / prod / f"{prod}_{idx}.py"

def check(job_id):
    p = py_for(job_id)
    if not p.exists():
        return (job_id, "?", "missing .py")
    try:
        _g, leaves = load_functional_source(p.read_text())
    except Exception as e:
        return (job_id, "?", f"load-fail {type(e).__name__}")
    flags = []
    for t in leaves:
        if not torch.is_tensor(t):
            flags.append("nontensor"); continue
        if t.is_floating_point():
            nf = (~torch.isfinite(t)).any().item()
            flags.append("NONFINITE" if nf else "finite")
        else:
            flags.append(f"int/{t.dtype}".replace("torch.", ""))
    leaf_nonfinite = any(f == "NONFINITE" for f in flags)
    return (job_id, "LEAF-NONFINITE" if leaf_nonfinite else "leaf-finite", ",".join(flags))

def main():
    ids = sys.argv[1:] or [l.strip() for l in sys.stdin if l.strip()]
    real = leafcaused = 0
    for jid in ids:
        job_id, verdict, detail = check(jid)
        print(f"{job_id}\t{verdict}\t{detail}")
        if verdict == "leaf-finite": real += 1
        elif verdict == "LEAF-NONFINITE": leafcaused += 1
    print(f"\n# {real} finite-leaf (op emits non-finite → REAL), "
          f"{leafcaused} leaf-already-nonfinite (filter per 3e)", file=sys.stderr)

if __name__ == "__main__":
    main()
