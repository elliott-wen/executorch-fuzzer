#!/usr/bin/env python3
"""Repro: aten.scatter.src_out does not write its source on the Vulkan delegate (Moto G54).

Phone-verified behavior: the device returns the input `self` with element 0 replaced by 0; the
source values are never written. Single-op graph -> no aliasing.

Phone-verified cases (Moto G54, 2026-06-29):
  scatter.src_out([1,2,3,4], 0, [0,2], [5,6])  eager [5,2,6,4]  device [0,2,3,4]
  scatter.src_out([0,0,0,0], 0, [0,2], [5,6])  eager [5,0,6,0]  device [0,0,0,0]

Needs a connected Vulkan worker (broker on 127.0.0.1:15554).
"""
import os, sys, time, warnings, logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
sys.path.insert(0, "/data/jwen929")
import zmq
from mobile.gen.export import build_job
from mobile.net import protocol as P, compare as cmp

SRC = """import torch
x = torch.tensor([1., 2., 3., 4.])
LEAVES = [x]
def g(x):
    idx = torch.tensor([0, 2], dtype=torch.int64); src = torch.tensor([5., 6.])
    return (torch.ops.aten.scatter.src_out(x, 0, idx, src, out=torch.empty(4)),)
"""

def run(src, name):
    job = build_job(src, "vulkan", False)
    assert job.status == "READY", getattr(job, "detail", job.status)
    ctx = zmq.Context.instance(); d = ctx.socket(zmq.DEALER); d.set_hwm(8)
    d.setsockopt(zmq.HEARTBEAT_IVL, 5000); d.setsockopt(zmq.HEARTBEAT_TIMEOUT, 20000)
    d.connect("tcp://127.0.0.1:15554"); pol = zmq.Poller(); pol.register(d, zmq.POLLIN)
    frames = P.encode_pushjob(f"repro::{name}", job.pte, job.inputs, job.eager,
                              desc=name, user_pos=job.user_pos)
    d.send_multipart(P.job_frames_from_pushjob(frames))
    t0 = time.monotonic()
    while time.monotonic() - t0 < 40:
        if dict(pol.poll(1000)).get(d):
            _, st, det, outs = P.decode_result(d.recv_multipart())
            if st != "RAN":
                print(f"device status={st} {det}"); return
            et = cmp.select(outs, job.user_pos)
            for a, b in zip(job.eager, et):
                eg = a.flatten().tolist(); dv = b.flatten().tolist()
                bad = "  <-- DIVERGES" if eg != dv else ""
                print(f"eager={eg}  device={dv}{bad}")
            return
    print("TIMEOUT")

if __name__ == "__main__":
    print("scatter.src_out([1,2,3,4], 0, index=[0,2], src=[5,6])")
    run(SRC, "scatter_src_drops_writes")
