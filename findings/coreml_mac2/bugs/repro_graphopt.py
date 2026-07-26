#!/usr/bin/env python3
"""repro_graphopt.py — A/B repro for a CoreML graph-optimization (SIBLING_DEPENDENT) bug.

Proves the target output is correct when returned ALONE but corrupted (ZEROED / SCALEd / …) once
the trigger sibling(s) are co-returned — i.e. a multi-output memory/quant-plan bug, not a kernel
bug. Lowers here (Linux, coreml), runs on the connected Mac worker via the broker.

  BROKER_PORT=15554 PYTHONPATH=/data/jwen929 .venv/bin/python repro_graphopt.py <job_id> <out_idx> <trigger_node>[,<n>...]
  e.g.  repro_graphopt.py w1:8 3 n3       # lift_fresh_copy zeroed by the `max` sibling
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import zmq
from mobile.gen.export import build_job
from mobile.net import protocol as P, compare as cmp, corpus as C
from mobile.gen.diff.cone_predicate import output_set_localizer

PORT = int(os.environ.get("BROKER_PORT", "15554"))
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER); D.connect(f"tcp://127.0.0.1:{PORT}")
POL = zmq.Poller(); POL.register(D, zmq.POLLIN); seq = [0]

def run(src):
    j = build_job(src, "coreml")
    if j.status != "READY": raise RuntimeError(f"BUILD:{j.status} {j.detail[:80]}")
    seq[0] += 1
    fr = P.encode_pushjob(f"rp::{seq[0]}", j.pte, j.inputs, j.eager, desc="rp", user_pos=j.user_pos)
    D.send_multipart(P.job_frames_from_pushjob(fr)); t0 = time.monotonic()
    while time.monotonic() - t0 < 90:
        if dict(POL.poll(1000)).get(D):
            _, st, det, raw = P.decode_result(D.recv_multipart())
            if st != "RAN": raise RuntimeError(f"DEV:{st}")
            et = cmp.select(raw, j.user_pos)
            if len(et) > len(j.eager): et = et[len(et) - len(j.eager):]
            return j.eager, et
    raise RuntimeError("DEV:TIMEOUT")

def main():
    job_id, out_idx = sys.argv[1], int(sys.argv[2])
    triggers = sys.argv[3].split(",") if len(sys.argv) > 3 else []
    others, build, tgt, ret = output_set_localizer(C.job_file("/data/jwen929/mobile/corpus_v4/coreml", job_id, "py"), out_idx)
    print(f"{job_id} out[{out_idx}] target={tgt}  outputs={ret}  triggers={triggers}")
    eg, et = run(build([tgt]))
    print(f"  A) [{tgt}] ALONE          : eager={eg[-1].flatten()[:6].tolist()}  device={et[-1].flatten()[:6].tolist()}")
    sel = [tgt, *triggers]
    eg, et = run(build(sel))
    i = [o for o in ret if o in set(sel)].index(tgt)
    ev, dv = eg[i].flatten()[:6].tolist(), et[i].flatten()[:6].tolist()
    print(f"  B) [{tgt}, {','.join(triggers)}] : eager={ev}  device={dv}")
    print(f"  => {'GRAPH-OPT CONFIRMED (correct alone, corrupted with sibling)' if dv != ev else 'no divergence'}")

if __name__ == "__main__":
    main()
