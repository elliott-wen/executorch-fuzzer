#!/usr/bin/env python3
"""Repro: Vulkan operator bug `_native_batch_norm_legit.no_stats` on Moto G54 5G (Android 14).
Single-op graph fed the op's EXACT runtime inputs (baked from corpus job w3:733 out[0]),
lowered to Vulkan, run on the connected device. Prints eager (CPU) vs device.
Needs a connected Vulkan worker (broker 127.0.0.1:15554).
"""
import os, sys, time, warnings, logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, "/data/jwen929")
import torch, zmq
from mobile.gen.export import build_job
from mobile.net import protocol as P, compare as cmp

SRC = 'import torch\ndef _first(r):\n    if isinstance(r, torch.Tensor): return r\n    if isinstance(r, (tuple, list)):\n        for x in r:\n            if isinstance(x, torch.Tensor): return x\n    return r\nB0 = torch.tensor([-0.1607666015625, 0.3701171875, 0.311279296875, 0.387939453125, 0.1712646484375, 0.2249755859375, 0.034210205078125, -0.045623779296875, -0.264404296875, -0.30419921875, -0.09295654296875, 0.26318359375], dtype=torch.float16).reshape([4, 3])\nLEAVES = [B0]\ndef g(B0):\n    T = _first(_first(torch.ops.aten._native_batch_norm_legit.no_stats(B0, None, None, True, 0.0, 0.0)))\n    return (T,)\n'

job = build_job(SRC, "vulkan", False)
assert job.status == "READY", f"lower failed: {job.status}"
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER)
D.setsockopt(zmq.HEARTBEAT_IVL, 5000); D.connect("tcp://127.0.0.1:15554")
P_ = zmq.Poller(); P_.register(D, zmq.POLLIN)
fr = P.encode_pushjob("repro::_native_batch_norm_legit_no_stats", job.pte, job.inputs, job.eager, desc="repro", user_pos=job.user_pos)
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
        print("op `_native_batch_norm_legit.no_stats`  ->", status, detail)
        print("  eager :", e.float().flatten()[:8].tolist())
        print("  device:", d.float().flatten()[:8].tolist())
        break
