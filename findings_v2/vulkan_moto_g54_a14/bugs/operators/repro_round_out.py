#!/usr/bin/env python3
"""Repro: Vulkan operator bug `round.out` on Moto G54 5G (Android 14).
Single-op graph fed the op's EXACT runtime inputs (baked from corpus job w101:24 out[0]),
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

SRC = 'import torch\ndef _first(r):\n    if isinstance(r, torch.Tensor): return r\n    if isinstance(r, (tuple, list)):\n        for x in r:\n            if isinstance(x, torch.Tensor): return x\n    return r\nB0 = torch.tensor([-0.4824913740158081, -0.4824913740158081, -0.4824913740158081, -0.4824913740158081, -0.4824913740158081, -0.4824913740158081, 1.1103534698486328, 1.1103534698486328, 1.1103534698486328, 1.1103534698486328, 1.1103534698486328, 1.1103534698486328, 0.9340156316757202, 0.9340156316757202, 0.9340156316757202, 0.9340156316757202, 0.9340156316757202, 0.9340156316757202, 1.1635862588882446, 1.1635862588882446, 1.1635862588882446, 1.1635862588882446, 1.1635862588882446, 1.1635862588882446, 0.513756513595581, 0.513756513595581, 0.513756513595581, 0.513756513595581, 0.513756513595581, 0.513756513595581, 0.6749985218048096, 0.6749985218048096, 0.6749985218048096, 0.6749985218048096, 0.6749985218048096, 0.6749985218048096], dtype=torch.float32).reshape([3, 2, 2, 3])\nLEAVES = [B0]\ndef g(B0):\n    T = _first(_first(torch.ops.aten.round.out(B0, out=torch.empty((3, 2, 2, 3), dtype=torch.float32))))\n    return (T,)\n'

job = build_job(SRC, "vulkan", False)
assert job.status == "READY", f"lower failed: {job.status}"
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER)
D.setsockopt(zmq.HEARTBEAT_IVL, 5000); D.connect("tcp://127.0.0.1:15554")
PL = zmq.Poller(); PL.register(D, zmq.POLLIN)
fr = P.encode_pushjob("repro::round_out", job.pte, job.inputs, job.eager, desc="repro", user_pos=job.user_pos)
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
        print("op `round.out`  ->", status, detail)
        ef, df = e.float().flatten(), d.float().flatten()
        if ef.numel() == df.numel() and ef.numel() > 0:
            diff = (ef - df).abs()
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
            w = int(diff.argmax())
            print(f"  worst element [idx {w}]:  eager={ef[w].item():.6g}  device={df[w].item():.6g}")
        print("  eager  head:", ef[:6].tolist())
        print("  device head:", df[:6].tolist())
        break
