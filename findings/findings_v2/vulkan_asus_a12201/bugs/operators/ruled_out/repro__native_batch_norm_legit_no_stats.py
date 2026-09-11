#!/usr/bin/env python3
"""Repro: Vulkan operator bug `_native_batch_norm_legit.no_stats` on ASUS a12201 (Vulkan).
Single-op graph fed the op's EXACT runtime inputs (baked from corpus job w44:561 out[0]),
lowered to Vulkan, run on the connected device. Prints eager (CPU) vs device.
Needs a connected Vulkan worker (broker 127.0.0.1:15554).
"""
import os, sys, time, warnings, logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, "/data/jwen929")
import torch, zmq
from mobile.gen.export import build_job
from mobile.executor import protocol as P, compare as cmp

SRC = 'import torch\ndef _first(r):\n    if isinstance(r, torch.Tensor): return r\n    if isinstance(r, (tuple, list)):\n        for x in r:\n            if isinstance(x, torch.Tensor): return x\n    return r\nB0 = torch.tensor([0.60302734375, -0.80615234375, 1.173828125, -0.8359375, -1.2451171875, 0.65185546875, -1.7216796875, 0.04534912109375, -4.39697265625, -5.80615234375, -3.826171875, -5.8359375, -1.2451171875, 0.65185546875, -1.7216796875, 0.04534912109375, 5.60302734375, 4.19384765625, 6.173828125, 4.1640625, -1.2451171875, 0.65185546875, -1.7216796875, 0.04534912109375, 5.60302734375, 4.19384765625, 6.173828125, 4.1640625, -1.2451171875, 0.65185546875, -1.7216796875, 0.04534912109375], dtype=torch.float32).reshape([1, 1, 4, 2, 4])\nLEAVES = [B0]\ndef g(B0):\n    T = _first(_first(torch.ops.aten._native_batch_norm_legit.no_stats(B0, None, None, True, 0.0, 0.0)))\n    return (T,)\n'

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
