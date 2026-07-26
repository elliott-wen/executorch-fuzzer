#!/usr/bin/env python3
"""repro_operator_bugs.py — VGF-delegate OPERATOR bugs (mismatch / operator).

Each op below computes a WRONG value when isolated STRICTLY ALONE on baked-correct constant inputs
and delegated to VGF (`bisect_cone` -> OPERATOR; delegated_ops>=1). Confirmed 5/5 deterministic by
the N-gate (findings/vgf/opgate.tsv). These are genuine per-op kernel bugs, not graph-opt.

The isolated single-op sub-graph is built with `cone_localizer(job, out).build_src([])` — the target
op alone, every input a constant baked from the eager reference — then run on VGF and diffed.

Needs the AoT env + broker + vgf_client worker (see repro_dropped_store.py header).
Run:
  PATH=/data/jwen929/mobile/.venv/bin:$PATH PYTHONPATH=/data/jwen929 \
  MODEL_CONVERTER_LIB_DIR=$(cat /data/jwen929/mobile/tmp/vgf_libdir.txt) BROKER_PORT=15774 \
  /data/jwen929/mobile/.venv/bin/python findings/vgf/bugs/repro_operator_bugs.py
"""
import os, sys, time
sys.path.insert(0, "/data/jwen929")
import torch, zmq
from mobile.gen.export import build_job
from mobile.net import protocol as P, compare as cmp, corpus as C
from mobile.gen.diff.cone_predicate import cone_localizer

CO = "/data/jwen929/mobile/corpus_v4/vgf"
PORT = int(os.environ.get("BROKER_PORT", "15774"))
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER); D.connect(f"tcp://127.0.0.1:{PORT}")
POL = zmq.Poller(); POL.register(D, zmq.POLLIN); seq = [0]

# (op, job, out_idx) — confirmed VGF_STABLE (5/5) operator bugs
CASES = [("native_group_norm", "w0:60", 1), ("floor_divide", "w100:219", 0),
         ("log1p", "w100:226", 1), ("cumsum", "w105:98", 4), ("asinh", "w18:636", 2),
         ("native_layer_norm", "w100:694", 0), ("sum", "w10:481", 2)]

def run(src):
    j = build_job(src, "vgf", False)
    if j.status != "READY":
        return None, None, j.status, -1
    seq[0] += 1
    fr = P.encode_pushjob(f"op::{seq[0]}", j.pte, j.inputs, j.eager, desc="op", user_pos=j.user_pos)
    D.send_multipart(P.job_frames_from_pushjob(fr)); t = time.monotonic()
    while time.monotonic() - t < 60:
        if dict(POL.poll(1000)).get(D):
            _, st, det, raw = P.decode_result(D.recv_multipart())
            et = cmp.select(raw, j.user_pos)
            if len(et) > len(j.eager): et = et[len(et) - len(j.eager):]
            return j.eager, et, st, j.delegated_ops
    return None, None, "TIMEOUT", -1

print("VGF-delegate OPERATOR bugs — isolated op alone on baked inputs, device vs eager\n" + "=" * 74)
for op, job, oi in CASES:
    anc, build_src, target = cone_localizer(C.job_file(CO, job, "py"), oi)
    ea, et, st, deleg = run(build_src([]))             # target alone, inputs baked to constants
    if et is None:
        print(f"\n{op:20s} {job} out[{oi}]  -> {st}"); continue
    ref, dev = ea[-1], et[-1]
    verdict, _ = cmp._cmp(ref, dev, 0)
    md = float((ref.flatten().float() - dev.flatten().float()).abs().max())
    print(f"\n{op:20s} {job} out[{oi}]  delegated_ops={deleg}")
    print(f"  eager ={ref.flatten()[:4].tolist()}")
    print(f"  device={dev.flatten()[:4].tolist()}")
    print(f"  {verdict}  max|delta|={md:.3e}")
