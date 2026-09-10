#!/usr/bin/env python3
"""Repro: Vulkan operator bug `narrow_copy` on Moto G54 5G (Android 14).
Single-op graph fed the op's EXACT runtime inputs (baked from corpus job w104:516 out[3]),
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

SRC = 'import torch\ndef _first(r):\n    if isinstance(r, torch.Tensor): return r\n    if isinstance(r, (tuple, list)):\n        for x in r:\n            if isinstance(x, torch.Tensor): return x\n    return r\nB0 = torch.tensor([-1.3411887884140015, -0.29623809456825256, 1.2547035217285156, 0.904482364654541, 0.004480193369090557, -0.09843862056732178, 2.125796318054199, -0.399491548538208, 1.3321086168289185, -1.448477029800415, -0.704727292060852, 0.6151601672172546, 0.10435733199119568, 0.9989657402038574, 0.15035493671894073, -0.6922657489776611, -1.1923149824142456, -0.9458473920822144, 0.5863195061683655, 1.0300838947296143, 0.9998095631599426, -0.8356595039367676, -0.9027174115180969, -0.8937790393829346, -0.25984248518943787, 1.4278374910354614, -1.065547227859497, 0.10348782688379288, 1.7545750141143799, -1.08524489402771, 0.243606299161911, -1.4743486642837524], dtype=torch.float32).reshape([2, 4, 4])\nLEAVES = [B0]\ndef g(B0):\n    T = _first(_first(torch.ops.aten.narrow_copy.default(B0, 0, 0, 2)))\n    return (T,)\n'

job = build_job(SRC, "vulkan", False)
assert job.status == "READY", f"lower failed: {job.status}"
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER)
D.setsockopt(zmq.HEARTBEAT_IVL, 5000); D.connect("tcp://127.0.0.1:15554")
P_ = zmq.Poller(); P_.register(D, zmq.POLLIN)
fr = P.encode_pushjob("repro::narrow_copy", job.pte, job.inputs, job.eager, desc="repro", user_pos=job.user_pos)
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
        print("op `narrow_copy`  ->", status, detail)
        print("  eager :", e.float().flatten()[:8].tolist())
        print("  device:", d.float().flatten()[:8].tolist())
        break
