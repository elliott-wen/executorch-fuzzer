#!/usr/bin/env python3
"""Batch non-OK classifier for cortex-m (analysis_single.md filters 3a + 3e), shardable.

For each non-OK job_id (from nonok_jobs.json keys op|status), emit a TSV row:
  job_id  op  status  cm_bucket  cm_compute  cm_all  leaf_nonfinite  note

  cm_bucket    (filter 3a, cortex-m analog): CM-COMPUTE | CM-QUANT-ONLY | NO-CM | LOWERFAIL
               — re-lower through the cortex-m backend and read cortex_m:: kernels from the .pte.
  leaf_nonfinite (filter 3e): a float input leaf already contains nan/inf (domain-fuzz injection).
               In a single-op graph nan/inf OUT where the leaf was finite is the op; a non-finite
               leaf makes the divergence input-handling, not a clean finite-domain kernel bug.

Usage: cm_classify.py --jobs nonok_jobs.json --out out.tsv --shard i/N
"""
import sys, os, io, contextlib, tempfile, json, argparse
from pathlib import Path

os.environ.setdefault("MOBILE_BACKENDS", "cortex-m")
sys.path.insert(0, "/data/jwen929")
ET_ROOT = "/data/jwen929/mobile/pytorch_ref/executorch"
sys.path.insert(0, ET_ROOT)
CORPUS = Path("/data/jwen929/mobile/corpus_v3/cortex-m")

import torch
from mobile.gen.export.job import build_job
from codegen.tools.gen_oplist import _get_operators

BOUNDARY = {"quantize_per_tensor", "dequantize_per_tensor"}
def _cmbase(o): return o.split("::", 1)[1].split(".")[0]

def jid_src(jid):
    w, n = jid.split(":")
    return (CORPUS / w / f"{w}_{n}.py").read_text()

def leaf_nonfinite(src):
    import runpy, tempfile as _tf
    with _tf.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); p = f.name
    try:
        ns = runpy.run_path(p)
        for t in ns.get("LEAVES", []):
            if t.dtype.is_floating_point and not bool(torch.isfinite(t).all()):
                return True
        return False
    except Exception:
        return None
    finally:
        os.unlink(p)

def cm_bucket(src):
    try:
        res = build_job(src, "cortex-m")
    except Exception as e:
        return "LOWERFAIL", [], [], f"exc:{type(e).__name__}"
    if res.status != "READY" or not res.pte:
        return "LOWERFAIL", [], [], f"{res.status}:{res.detail[:50]}"
    with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
        f.write(res.pte); pte = f.name
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ops = sorted(_get_operators(pte))
    finally:
        os.unlink(pte)
    cm = [o for o in ops if o.startswith("cortex_m::")]
    comp = [o for o in cm if _cmbase(o) not in BOUNDARY]
    b = "CM-COMPUTE" if comp else ("CM-QUANT-ONLY" if cm else "NO-CM")
    return b, comp, cm, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    i, N = map(int, a.shard.split("/"))
    d = json.loads(Path(a.jobs).read_text())
    work = []
    for key, jids in d.items():
        op, status = key.split("|")
        for jid in jids:
            work.append((jid, op, status))
    work = [w for idx, w in enumerate(sorted(work)) if idx % N == i]
    with open(a.out, "w") as out:
        for jid, op, status in work:
            try:
                src = jid_src(jid)
            except Exception as e:
                out.write(f"{jid}\t{op}\t{status}\tSRCFAIL\t\t\t\t{e}\n"); out.flush(); continue
            nf = leaf_nonfinite(src)
            b, comp, cm, note = cm_bucket(src)
            out.write(f"{jid}\t{op}\t{status}\t{b}\t{','.join(_cmbase(o) for o in comp)}\t"
                      f"{','.join(_cmbase(o) for o in cm)}\t{nf}\t{note}\n")
            out.flush()

if __name__ == "__main__":
    main()
