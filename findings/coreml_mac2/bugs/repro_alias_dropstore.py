#!/usr/bin/env python3
"""repro_alias_dropstore.py — MINIMAL self-contained repro of the CoreML dropped-store graph-opt bug.

Two returned outputs that alias the SAME tensor: on CoreML the 2nd reads back all-zeros (the store is
dropped); on CPU (portable) both are correct. No corpus job needed. Lowers here (Linux, coreml), runs
on the connected Mac worker via the broker.

  BROKER_PORT=15554 PYTHONPATH=/data/jwen929 /data/jwen929/mobile/.venv/bin/python repro_alias_dropstore.py
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import zmq
from mobile.gen.export import build_job
from mobile.net import protocol as P

SRC = '''import torch
torch.manual_seed(0)
L0 = torch.randn(4)
def g(L0):
    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)
    a  = torch.ops.aten.lift_fresh_copy.default(n2)   # view/identity of n2
    b  = torch.ops.aten.alias_copy.default(n2)        # another alias of the SAME n2
    return (a, b)
LEAVES = [L0]
'''

def run(src, backend):
    j = build_job(src, backend)
    if j.status != "READY": return j.status, None, None
    if backend != "coreml":
        from mobile.net.et_runner import run_pte
        out = run_pte(j.pte, j.inputs)
        return "RAN", [t.flatten().tolist() for t in j.eager], [o.flatten().tolist() for o in out]
    ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER); D.connect(f"tcp://127.0.0.1:{os.environ.get('BROKER_PORT','15554')}")
    POL = zmq.Poller(); POL.register(D, zmq.POLLIN)
    fr = P.encode_pushjob("alias::1", j.pte, j.inputs, j.eager, desc="alias", user_pos=j.user_pos)
    D.send_multipart(P.job_frames_from_pushjob(fr)); t0 = time.monotonic()
    while time.monotonic() - t0 < 90:
        if dict(POL.poll(1000)).get(D):
            _, st, det, raw = P.decode_result(D.recv_multipart())
            if st != "RAN": return st, None, None
            return "RAN", [t.flatten().tolist() for t in j.eager], [r.flatten().tolist() for r in raw]
    return "TIMEOUT", None, None

if __name__ == "__main__":
    for be in ("portable", "coreml"):
        st, eg, dv = run(SRC, be)
        print(f"[{be}] {st}")
        if st == "RAN":
            for i, (e, d) in enumerate(zip(eg, dv)):
                z = " <-- ZEROED (store dropped)" if all(abs(x) < 1e-9 for x in d) and any(abs(x) > 1e-6 for x in e) else ""
                print(f"    out[{i}] eager={[round(x,3) for x in e]} device={[round(x,3) for x in d]}{z}")
