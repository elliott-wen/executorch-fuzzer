#!/usr/bin/env python3
"""Coverage ledger (analysis_single.md step 2/coverage gate) for cortex-m:
scheduled (EXECUTORCH_OPS) vs produced (in corpus) vs never-produced gap, plus
generation-time attrition from corpus_v3/cortex-m/_crashes/*.py.

cortex-m is pass-based (no delegate), so there is no delegated-vs-portable split at the
job header. Instead we tag each produced op by whether the CortexM pass *can* rewrite it
into a cortex_m:: CMSIS-NN compute kernel (CM_COMPUTE_OPS, the static backend surface).
Every quantized graph additionally exercises the cortex_m quantize/dequantize boundary
kernels regardless of its compute op.
"""
import re, collections
from pathlib import Path
import mobile.gen.ops.allowlist as A

CORPUS = Path("/data/jwen929/mobile/corpus_v3/cortex-m")
MAN = Path("/data/jwen929/mobile/findings/cortex_m/_work/manifest.tsv")

# aten source ops the CortexM passes lower into cortex_m:: compute kernels (backend surface).
# Derived from backends/cortex_m/passes/*  (ConvertToCortexMPass + fusion/activation passes).
CM_COMPUTE_ATEN = {
    "linear", "addmm", "mm", "matmul", "bmm",
    "convolution", "conv2d", "conv1d", "conv_transpose2d",
    "add", "add.Tensor", "add.Scalar", "add.out",
    "mul", "mul.Tensor", "mul.Scalar", "mul.out",
    "maximum", "minimum", "maximum.out", "minimum.out",
    "avg_pool2d", "max_pool2d", "_adaptive_avg_pool2d", "mean", "mean.dim",
    "softmax", "_softmax", "_safe_softmax",
    "hardswish", "hardsigmoid", "relu", "clamp", "hardtanh",
    "permute", "permute_copy", "transpose", "transpose_copy", "t", "t_copy",
    "constant_pad_nd", "pad",
}

def base_op(op: str) -> str:
    return op.split(".")[0]

scheduled = set(A.EXECUTORCH_OPS)
produced = set()
op_count = collections.Counter()
for ln in MAN.read_text().splitlines()[1:]:
    f = ln.split("\t")
    if len(f) < 2:
        continue
    op = f[1]
    produced.add(op); op_count[op] += 1

never = scheduled - produced
extra = produced - scheduled

op_re = re.compile(r"Graph:\s*n0\*?=([a-zA-Z0-9_.]+)\(")
gen_crash_ops = collections.Counter()
for py in CORPUS.glob("_crashes/*.py"):
    try:
        txt = py.read_text(errors="replace")
    except Exception:
        continue
    m = op_re.search(txt)
    if m:
        gen_crash_ops[m.group(1)] += 1

cm_capable = sorted(o for o in produced if base_op(o) in CM_COMPUTE_ATEN or o in CM_COMPUTE_ATEN)
print(f"SCHEDULED (EXECUTORCH_OPS): {len(scheduled)}")
print(f"PRODUCED (in corpus):       {len(produced)}")
print(f"  of which cortex_m COMPUTE-capable (backend surface): {len(cm_capable)}")
print(f"  (all produced graphs also run cortex_m quantize/dequantize boundary kernels)")
print(f"NEVER PRODUCED (gap):       {len(never)}")
print(f"PRODUCED-but-not-scheduled (decomp/alias names): {len(extra)}")
print()
print(f"ASSERT scheduled == produced_in_schedule + never : "
      f"{len(scheduled & produced)} + {len(never)} = {len(scheduled & produced)+len(never)} vs {len(scheduled)}")
print()
print("== NEVER-PRODUCED ops (coverage gaps — scheduled but never lowered) ==")
for op in sorted(never):
    gc = gen_crash_ops.get(op, 0)
    tag = f"gen-crash x{gc}" if gc else "no job produced (seed-infeasible / partition/lowering attrition)"
    print(f"  {op:<40} {tag}")
print()
print(f"== generation-time _crashes: {sum(gen_crash_ops.values())} graphs over {len(gen_crash_ops)} ops ==")
for op, c in gen_crash_ops.most_common(40):
    print(f"  {op:<40} x{c:<5} {'PRODUCED-elsewhere' if op in produced else 'NEVER-produced'}")
print()
print("== cortex_m COMPUTE-capable produced ops (the backend compute surface) ==")
print(f"  ({len(cm_capable)} ops) " + ", ".join(cm_capable))
