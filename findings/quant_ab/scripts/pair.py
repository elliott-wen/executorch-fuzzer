#!/usr/bin/env python3
"""Build two symlink corpora restricted to the job slots present in BOTH.

Same seed schedule ⇒ job key 'w7_412' is the SAME generated graph in the float and the
int8 corpus. Feeding only the intersection makes the runtime A/B strictly paired: every
graph is compared against itself, so a rate difference can't come from corpus drift.

usage: pair.py <float_dir> <int8_dir> <out_float> <out_int8>
"""
import sys, pathlib, os
fl, q8, of, oq = (pathlib.Path(p) for p in sys.argv[1:5])
A = {p.stem for p in fl.glob("*/*.job")}
B = {p.stem for p in q8.glob("*/*.job")}
both = sorted(A & B)
for src, dst in ((fl, of), (q8, oq)):
    if dst.exists():
        import shutil; shutil.rmtree(dst)
    for k in both:
        w = k.split("_")[0]
        (dst / w).mkdir(parents=True, exist_ok=True)
        for ext in ("job", "py"):
            s = src / w / f"{k}.{ext}"
            if s.exists():
                os.symlink(s.resolve(), dst / w / f"{k}.{ext}")
print(f"float={len(A)} int8={len(B)} paired={len(both)} -> {of} , {oq}")
