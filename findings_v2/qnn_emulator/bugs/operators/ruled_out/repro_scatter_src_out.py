#!/usr/bin/env python3
"""Repro: Vulkan operator bug `scatter.src_out` on QNN HTP x86 emulator (fp16).
Single-op graph fed the op's EXACT runtime inputs (baked from corpus job w10:379 out[0]),
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

SRC = 'import torch\ndef _first(r):\n    if isinstance(r, torch.Tensor): return r\n    if isinstance(r, (tuple, list)):\n        for x in r:\n            if isinstance(x, torch.Tensor): return x\n    return r\nB0 = torch.tensor([-0.9591930508613586, -0.9591930508613586, 0.9999999403953552, 1.0000025033950806, -0.9999999403953552, -0.9999973773956299, 0.9999999403953552, 1.0000025033950806, -0.9999999403953552, -0.9999973773956299, 0.9999999403953552, 1.0000025033950806, -0.3056862950325012, -0.3056862950325012, 0.9999999403953552, 1.0000025033950806, 2.4141592979431152, 2.4141592979431152, 2.4141592979431152, 2.4141592979431152, -0.2841799557209015, -0.2841799557209015, 0.9999999403953552, 1.0000025033950806, -0.013869979418814182, -0.013869979418814182, 0.9999999403953552, 1.0000025033950806, 0.8924581408500671, 0.8924581408500671, 0.9999999403953552, 1.0000025033950806, 0.28180453181266785, 0.28180453181266785, 0.9999999403953552, 1.0000025033950806, -0.529233992099762, -0.529233992099762, 0.9999999403953552, 1.0000025033950806, -0.874897837638855, -0.874897837638855, 0.9999999403953552, 1.0000025033950806, 0.0920817106962204, 0.0920817106962204, 0.9999999403953552, 1.0000025033950806, -0.9999999403953552, -0.9999973773956299, 0.9999999403953552, 1.0000025033950806, -0.6797337532043457, -0.6797337532043457, 0.9999999403953552, 1.0000025033950806, 0.05864547938108444, 0.05864547938108444, 0.9999999403953552, 1.0000025033950806, -0.9999999403953552, -0.9999973773956299, 0.9999999403953552, 1.0000025033950806], dtype=torch.float32).reshape([1, 4, 4, 2, 2])\nB1 = torch.tensor([0], dtype=torch.int64).reshape([1, 1, 1, 1, 1])\nB2 = torch.tensor([-3.1989028453826904], dtype=torch.float32).reshape([1, 1, 1, 1, 1])\nLEAVES = [B0, B1, B2]\ndef g(B0, B1, B2):\n    T = _first(_first(torch.ops.aten.scatter.src_out(B0, 0, B1, B2, out=torch.empty((1, 4, 4, 2, 2), dtype=torch.float32))))\n    return (T,)\n'

job = build_job(SRC, "qualcomm", False)
assert job.status == "READY", f"lower failed: {job.status}"
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER)
D.setsockopt(zmq.HEARTBEAT_IVL, 5000); D.connect("tcp://127.0.0.1:15564")
P_ = zmq.Poller(); P_.register(D, zmq.POLLIN)
fr = P.encode_pushjob("repro::scatter_src_out", job.pte, job.inputs, job.eager, desc="repro", user_pos=job.user_pos)
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
        print("op `scatter.src_out`  ->", status, detail)
        print("  eager :", e.float().flatten()[:8].tolist())
        print("  device:", d.float().flatten()[:8].tolist())
        break
