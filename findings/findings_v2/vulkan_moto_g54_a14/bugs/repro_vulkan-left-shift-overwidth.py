#!/usr/bin/env python3
"""Repro: aten.bitwise_left_shift over-width shift diverges on the Vulkan delegate (Moto G54).

For an int32 tensor, `x << 40` (shift >= bit width 32):
  - eager (CPU / PyTorch): result is 0  (over-width shift defined as 0)
  - Vulkan/Mali GPU: shifts by (40 mod 32) = 8, giving x*256
Single-op graph -> no aliasing; pure kernel/semantics divergence.

Needs a connected Vulkan worker (broker on 127.0.0.1:15554).
"""
import os, sys, time, warnings, logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
sys.path.insert(0, "/data/jwen929")
import zmq
from mobile.gen.export import build_job
from mobile.executor import protocol as P, compare as cmp

SRC = """import torch
x = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
LEAVES = [x]
def g(x):
    return (torch.ops.aten.bitwise_left_shift.Tensor_Scalar(x, 40),)
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
                eg = a.flatten().tolist(); dv = b.flatten().tolist()
                bad = "  <-- DIVERGES" if eg != dv else ""
                print(f"out[{i}] eager={eg}\n       device={dv}{bad}")
            return
    print("TIMEOUT")

if __name__ == "__main__":
    print("int32 [1,2,3,4] << 40  (shift >= bit width)")
    run(SRC, "left_shift_overwidth")
