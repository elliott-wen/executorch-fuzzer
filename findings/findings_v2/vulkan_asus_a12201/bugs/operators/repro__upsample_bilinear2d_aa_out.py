#!/usr/bin/env python3
"""Repro: Vulkan operator bug `_upsample_bilinear2d_aa.out` on Moto G54 5G (Android 14).
Single-op graph fed the op's EXACT runtime inputs (baked from corpus job w16:743 out[1]),
lowered to Vulkan, run on the connected device. Prints eager vs device at the worst element.
Needs a connected Vulkan worker (broker 127.0.0.1:15554).
"""
import os, sys, time, warnings, logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, "/data/jwen929")
import torch, zmq
from mobile.gen.export import build_job
from mobile.net import protocol as P, compare as cmp

SRC = 'import torch\ndef _first(r):\n    if isinstance(r, torch.Tensor): return r\n    if isinstance(r, (tuple, list)):\n        for x in r:\n            if isinstance(x, torch.Tensor): return x\n    return r\nB0 = torch.tensor([6, 6, 6, 248, 6, 2, 8, 4, 6, 2, 248, 2, 248, 250, 6, 8, 6, 6, 254, 6, 252, 248, 254, 0, 254, 254, 4, 4, 0, 254, 252, 2, 6, 8, 252, 248, 4, 8, 2, 2, 248, 6, 2, 8, 248, 254, 2, 4, 248, 252, 250, 0, 8, 6, 250, 254, 2, 250, 248, 254, 8, 252, 254, 248, 2, 2, 252, 254, 0, 250, 254, 4, 254, 8, 250, 2, 4, 252, 8, 252, 4, 252, 6, 0, 248, 2, 252, 254, 0, 252, 254, 8, 2, 4, 2, 6], dtype=torch.uint8).reshape([4, 2, 4, 3])\nLEAVES = [B0]\ndef g(B0):\n    T = _first(_first(torch.ops.aten._upsample_bilinear2d_aa.out(B0, [3, 2], True, None, None, out=torch.empty((4, 2, 3, 2), dtype=torch.uint8))))\n    return (T,)\n'

job = build_job(SRC, "vulkan", False)
assert job.status == "READY", f"lower failed: {job.status}"
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER)
D.setsockopt(zmq.HEARTBEAT_IVL, 5000); D.connect("tcp://127.0.0.1:15554")
PL = zmq.Poller(); PL.register(D, zmq.POLLIN)
fr = P.encode_pushjob("repro::_upsample_bilinear2d_aa_out", job.pte, job.inputs, job.eager, desc="repro", user_pos=job.user_pos)
D.send_multipart(P.job_frames_from_pushjob(fr))
t0 = time.monotonic()
while time.monotonic() - t0 < 60:
    if dict(PL.poll(1000)).get(D):
        _, st, det, raw = P.decode_result(D.recv_multipart())
        if st != "RAN":
            print("device status:", st, det[:120]); break
        et = cmp.select(raw, job.user_pos)
        if len(et) > len(job.eager): et = et[len(et)-len(job.eager):]
        e, d = job.eager[0], et[0]
        status, detail = cmp._cmp(e, d, 0)
        print("op `_upsample_bilinear2d_aa.out`  ->", status, detail)
        ef, df = e.float().flatten(), d.float().flatten()
        if ef.numel() == df.numel() and ef.numel() > 0:
            diff = (ef - df).abs()
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
            w = int(diff.argmax())
            print(f"  worst element [idx {w}]:  eager={ef[w].item():.6g}  device={df[w].item():.6g}")
        print("  eager  head:", ef[:6].tolist())
        print("  device head:", df[:6].tolist())
        break
