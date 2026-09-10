#!/usr/bin/env python3
"""Per-op ANNOTATION RATE over the REAL corpus (not one representative graph).

qsweep.py answers "does the quantizer annotate this op" from ONE sample graph per op.
Annotation depends on dtype/shape, so that understates coverage. This samples K real
single-op jobs per op from a corpus and reports how many get q/dq inserted.

usage: ann_rate.py <backend> <corpus_dir> <out.tsv> [K]
"""
import os, sys, re, collections, pathlib, random
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from torch.export import export
from mobile.gen.export.job import load_functional_source, _GraphModule
from mobile.gen.export.backends import get_backend
from mobile.gen.export.backends.base import _convert_pt2e_module

BK, CORPUS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 3
bk = get_backend(BK)
if bk is None: sys.exit(f"backend {BK} unavailable")
QOPS = ("quantize_per_tensor", "dequantize_per_tensor", "quantize_per_channel", "dequantize_per_channel")
RE_G = re.compile(r"Graph: (.*)"); RE_OP = re.compile(r"n\d+\*?=([A-Za-z_][\w.]*)\(")

by_op = collections.defaultdict(list)
for p in pathlib.Path(CORPUS).glob("*/*.py"):
    head = p.read_text(errors="replace")[:700]
    g = RE_G.search(head)
    if not g: continue
    ops = RE_OP.findall(g.group(1))
    if len(ops) == 1: by_op[ops[0]].append(p)

rng = random.Random(0)
# resume: a hostile SDK (QNN) can hard-crash the process mid-sweep — keep finished rows.
done = set()
if os.path.exists(OUT):
    done = {l.split("\t")[0] for l in open(OUT).read().splitlines()[1:]}
fh = open(OUT, "a" if done else "w")
if not done: fh.write("op\tsamples\tannotated\tmean_qdq\terrs\n")
for i, (op, paths) in enumerate(sorted(by_op.items()), 1):
    if op in done: continue
    sel = rng.sample(paths, min(K, len(paths)))
    ann = errs = 0; qs = []
    for p in sel:
        try:
            src = p.read_text()
            g, leaves = load_functional_source(src)
            ep = export(_GraphModule(g).eval(), tuple(t.clone() for t in leaves))
            conv = _convert_pt2e_module(ep, tuple(t.clone() for t in leaves), bk.quantizer())
            nq = sum(1 for n in conv.graph.nodes
                     if n.op == "call_function" and any(k in str(n.target) for k in QOPS))
            qs.append(nq); ann += nq > 0
        except Exception:
            errs += 1
    fh.write(f"{op}\t{len(sel)}\t{ann}\t{round(sum(qs)/len(qs),2) if qs else 0}\t{errs}\n"); fh.flush()
    if i % 25 == 0: print(f"  [{BK}] {i}/{len(by_op)}", file=sys.stderr, flush=True)
fh.close()
rows = [l.split("\t") for l in open(OUT).read().splitlines()[1:]]
full = sum(1 for r in rows if int(r[2]) == int(r[1]) and int(r[2]) > 0)
part = sum(1 for r in rows if 0 < int(r[2]) < int(r[1]))
none = sum(1 for r in rows if int(r[2]) == 0)
print(f"\n== {BK}: ops={len(rows)}  always-annotated={full}  sometimes={part}  never={none}")
