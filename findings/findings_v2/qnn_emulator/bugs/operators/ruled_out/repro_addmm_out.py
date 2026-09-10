#!/usr/bin/env python3
"""Repro: Vulkan operator bug `addmm.out` on QNN HTP x86 emulator (fp16).
Single-op graph fed the op's EXACT runtime inputs (baked from corpus job w101:508 out[0]),
lowered to Vulkan, run on the connected device. Prints eager (CPU) vs device.
Needs a connected Vulkan worker (broker 127.0.0.1:15564).
"""
import os, sys, time, warnings, logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, "/data/jwen929")
import torch, zmq
from mobile.gen.export import build_job
from mobile.net import protocol as P, compare as cmp

SRC = 'import torch\ndef _first(r):\n    if isinstance(r, torch.Tensor): return r\n    if isinstance(r, (tuple, list)):\n        for x in r:\n            if isinstance(x, torch.Tensor): return x\n    return r\nB0 = torch.tensor([15.0, 15.0], dtype=torch.float32).reshape([2])\nB1 = torch.tensor([-0.1953093260526657, 3.0123977661132812, 1.0365084409713745], dtype=torch.float32).reshape([3, 1])\nB2 = torch.tensor([0.0, 0.0], dtype=torch.float32).reshape([1, 2])\nLEAVES = [B0, B1, B2]\ndef g(B0, B1, B2):\n    T = _first(_first(torch.ops.aten.addmm.out(B0, B1, B2, beta=-3, alpha=-6, out=torch.empty((3, 2), dtype=torch.float32))))\n    return (T,)\n'

job = build_job(SRC, "qualcomm", False)
assert job.status == "READY", f"lower failed: {job.status}"
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER)
D.setsockopt(zmq.HEARTBEAT_IVL, 5000); D.connect("tcp://127.0.0.1:15564")
P_ = zmq.Poller(); P_.register(D, zmq.POLLIN)
fr = P.encode_pushjob("repro::addmm_out", job.pte, job.inputs, job.eager, desc="repro", user_pos=job.user_pos)
D.send_multipart(P.job_frames_from_pushjob(fr))
t0 = time.monotonic()
while time.monotonic() - t0 < 60:
    if dict(P_.poll(1000)).get(D):
        _, st, det, raw = P.decode_result(D.recv_multipart())
        if st != "RAN":
            print("device status:", st, det[:120]); break
        et = cmp.select(raw, job.user_pos)
        if len(et) > len(job.eager): et = et[len(et)-len(job.eager):]
        e, d = job.eager[0], et[0]
        status, detail = cmp._cmp(e, d, 0)
        print("op `addmm.out`  ->", status, detail)
        print("  eager :", e.float().flatten()[:8].tolist())
        print("  device:", d.float().flatten()[:8].tolist())
        break
