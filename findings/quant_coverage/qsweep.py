#!/usr/bin/env python3
"""Per-op quantization coverage for one backend.

For each representative single-op graph:
  ANNOTATED  - the backend's PT2E quantizer inserted q/dq around the op (convert_pt2e)
  LOWERED    - build_job(..., quantize=True) returns READY
  DELEGATED  - that quantized job has delegated_ops >= 1

ANNOTATED is the real "can this op be quantized" answer and is independent of whether the
backend's quantized *lowering* works (e.g. qualcomm's is broken but annotates fine).

usage: qsweep.py <backend> <out.tsv> [--limit N] [--no-lower]
"""
import os, sys, pathlib, traceback
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from torch.export import export
from mobile.gen.export.job import load_functional_source, _GraphModule
from mobile.gen.export.backends import get_backend
from mobile.gen.export.backends.base import _convert_pt2e_module
from mobile.gen.export import build_job

BK_NAME = sys.argv[1]
OUT = sys.argv[2]
LIMIT = int(sys.argv[sys.argv.index("--limit")+1]) if "--limit" in sys.argv else 0
DO_LOWER = "--no-lower" not in sys.argv

bk = get_backend(BK_NAME)
if bk is None:
    sys.exit(f"backend {BK_NAME!r} unavailable in this env")
q = bk.quantizer()
if q is None:
    sys.exit(f"backend {BK_NAME!r} has no quantizer (QuantMode={bk.quant})")

QOPS = ("quantize_per_tensor", "dequantize_per_tensor",
        "quantize_per_channel", "dequantize_per_channel")

rows = [l.rstrip("\n").split("\t") for l in open("oplist.tsv")]
if LIMIT: rows = rows[:LIMIT]

fh = open(OUT, "w")
fh.write("op\tannotated\tn_qdq\tlowered\tdeleg_ops\tnote\n")
for i, (op, path) in enumerate(rows, 1):
    ann, nq, low, deleg, note = "?", 0, "-", "-", ""
    try:
        src = open(path).read()
        g, leaves = load_functional_source(src)
        ep = export(_GraphModule(g).eval(), tuple(t.clone() for t in leaves))
        conv = _convert_pt2e_module(ep, tuple(t.clone() for t in leaves), bk.quantizer())
        nq = sum(1 for n in conv.graph.nodes
                 if n.op == "call_function" and any(k in str(n.target) for k in QOPS))
        ann = "YES" if nq > 0 else "no"
    except Exception as e:
        ann, note = "ERR", f"{type(e).__name__}: {str(e)[:80]}"
    if DO_LOWER and ann == "YES":
        try:
            r = build_job(src, BK_NAME, True)
            low = r.status
            deleg = r.delegated_ops if r.status == "READY" else "-"
            if r.status != "READY": note = (note + " | " + r.detail[:70]).strip(" |")
        except Exception as e:
            low, note = "EXC", (note + " | " + f"{type(e).__name__}: {str(e)[:60]}").strip(" |")
    fh.write(f"{op}\t{ann}\t{nq}\t{low}\t{deleg}\t{note}\n"); fh.flush()
    if i % 25 == 0: print(f"  [{BK_NAME}] {i}/{len(rows)}", file=sys.stderr, flush=True)
fh.close()

# summary
import collections
c = collections.Counter(l.split("\t")[1] for l in open(OUT).read().splitlines()[1:])
d = [l.split("\t") for l in open(OUT).read().splitlines()[1:]]
print(f"\n== {BK_NAME}: ops={len(d)}  ANNOTATED={c.get('YES',0)}  not={c.get('no',0)}  ERR={c.get('ERR',0)}")
if DO_LOWER:
    ready = sum(1 for r in d if r[3] == "READY")
    dele  = sum(1 for r in d if r[3] == "READY" and r[4] not in ("-", "0"))
    print(f"   of ANNOTATED: LOWERED READY={ready}  DELEGATED(ops>=1)={dele}")
