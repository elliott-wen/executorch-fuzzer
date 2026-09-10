#!/usr/bin/env python3
"""Repro: aten.floor_divide(x, x) returns 0 instead of 1 on the Vulkan delegate (Moto G54).

floor_divide(x, x) == floor(x/x) == floor(1.0) == 1 for all x != 0. On the Mali GPU the
division x/x rounds to 0.99999... and floor() then yields 0. Eager (CPU) computes x/x as
exactly 1.0. Single-op graph -> no aliasing involved; this is a pure kernel divergence.

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
x = torch.tensor([0.3, 0.7, 1.1, 1.7, 2.3, 2.9, 3.3, 3.7])
LEAVES = [x]
def g(x):
    return (torch.ops.aten.floor_divide.default(x, x),)
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
            for i, (a, b) in enumerate(zip(job.eager, et)):
                eg = [round(v, 4) for v in a.flatten().tolist()]
                dv = [round(v, 4) for v in b.flatten().tolist()]
                bad = "  <-- DIVERGES" if eg != dv else ""
                print(f"out[{i}] eager={eg}\n       device={dv}{bad}")
            return
    print("TIMEOUT")

if __name__ == "__main__":
    print("floor_divide(x, x)  for x = [0.3, 0.7, 1.1, 1.7, 2.3, 2.9, 3.3, 3.7]")
    run(SRC, "floor_divide_self")
