#!/usr/bin/env python3
"""Filter-3a replacement for cortex-m (analysis_single.md 3a): "whose kernel ran?".

cortex-m is pass-based, so the .job delegated.ops counter is always 0 and useless. The
real signal is which cortex_m:: CMSIS-NN kernels the LOWERED program actually calls. We
re-lower each job through the cortex-m backend (the exact build_job the corpus used) and
read the registered operator names out of the resulting .pte with executorch's gen_oplist
(_get_operators) — the same probe fvp_runner.sh uses to pick kernels.

Per job we emit: job_id, op, cm_compute (cortex_m:: compute kernels present, excluding the
ubiquitous quantize/dequantize boundary), all_cm (every cortex_m:: op), n_portable_aten.

Bucket:
  CM-COMPUTE : the op-under-test became a cortex_m:: compute kernel -> in-scope cortex-m bug.
  CM-QUANT-ONLY : only cortex_m quantize/dequantize present; the compute op ran as a portable
                  quantized_decomposed/aten kernel -> a divergence is portable/boundary, not a
                  cortex-m compute-kernel bug (still flag if it's the shared quant kernel).
  NO-CM : no cortex_m op at all (lowering didn't quantize) -> out of scope for cortex-m.

Usage: cm_kernels.py <job_id> [job_id ...]   (or --tsv <skip.tsv> to process all non-OK)
"""
import sys, os, tempfile, io, contextlib, json
from pathlib import Path

os.environ.setdefault("MOBILE_BACKENDS", "cortex-m")
sys.path.insert(0, "/data/jwen929")
CORPUS = Path("/data/jwen929/mobile/corpus_v3/cortex-m")

from mobile.gen.export.job import build_job

ET_ROOT = "/data/jwen929/mobile/pytorch_ref/executorch"
sys.path.insert(0, ET_ROOT)
from codegen.tools.gen_oplist import _get_operators

def jid_to_src(jid):
    w, n = jid.split(":")
    return (CORPUS / w / f"{w}_{n}.py").read_text()

def cm_ops_of(jid):
    src = jid_to_src(jid)
    res = build_job(src, "cortex-m")
    if res.status != "READY" or not res.pte:
        return None, f"lower-{res.status}: {res.detail[:80]}"
    with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
        f.write(res.pte); pte = f.name
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ops = sorted(_get_operators(pte))
    finally:
        os.unlink(pte)
    return ops, ""

def _cmbase(o):   # cortex_m::quantize_per_tensor.out -> quantize_per_tensor
    return o.split("::", 1)[1].split(".")[0]

BOUNDARY = {"quantize_per_tensor", "dequantize_per_tensor"}

def classify(ops):
    cm = [o for o in ops if o.startswith("cortex_m::")]
    cm_compute = [o for o in cm if _cmbase(o) not in BOUNDARY]
    if cm_compute:
        return "CM-COMPUTE", cm, cm_compute
    if cm:
        return "CM-QUANT-ONLY", cm, cm_compute
    return "NO-CM", cm, cm_compute

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--tsv":
        jids = [ln.split("\t")[1] for ln in Path(args[1]).read_text().splitlines()[1:] if "\t" in ln]
    else:
        jids = args
    print(f"{'job_id':<12}{'bucket':<15}{'cm_compute':<40}{'#cm':>4}")
    for jid in jids:
        try:
            ops, err = cm_ops_of(jid)
        except Exception as e:
            print(f"{jid:<12}{'ERROR':<15}{str(e)[:50]}"); continue
        if ops is None:
            print(f"{jid:<12}{'LOWERFAIL':<15}{err}"); continue
        bucket, cm, cm_compute = classify(ops)
        print(f"{jid:<12}{bucket:<15}{','.join(o.split('::')[1] for o in cm_compute)[:39]:<40}{len(cm):>4}")
