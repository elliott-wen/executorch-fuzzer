#!/usr/bin/env python3
"""Repro: Vulkan operator bug `gelu.out` on Moto G54 5G (Android 14).
Single-op graph fed the op's EXACT runtime inputs (baked from corpus job w115:120 out[0]),
lowered to Vulkan, run on the connected device. Prints eager vs device at the worst element.
Needs the QNN emulator: source tmp/qnn_env.sh first, and a QNN client on broker 127.0.0.1:15564.
"""
import os, sys, time, warnings, logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, "/data/jwen929")
import torch, zmq
from mobile.gen.export import build_job
from mobile.net import protocol as P, compare as cmp

SRC = "import torch\ndef _first(r):\n    if isinstance(r, torch.Tensor): return r\n    if isinstance(r, (tuple, list)):\n        for x in r:\n            if isinstance(x, torch.Tensor): return x\n    return r\nB0 = torch.tensor([1.6701700587873347e-05, 0.0009118819725699723, 0.00012340980174485594, 7.389056205749512, 148.4131622314453, 0.049787066876888275, 148.4131622314453, 20.08553695678711, 0.00012340980174485594, 0.018315639346837997, 0.00012340980174485594, 0.0009118819725699723, 0.3678794503211975, 0.00012340980174485594, 2.7182817459106445, 162754.796875, 0.3678794503211975, 0.3678794503211975], dtype=torch.float32).reshape([3, 3, 2])\nLEAVES = [B0]\ndef g(B0):\n    T = _first(_first(torch.ops.aten.gelu.out(B0, approximate='none', out=torch.empty((3, 3, 2), dtype=torch.float32))))\n    return (T,)\n"

job = build_job(SRC, "qualcomm", False)
assert job.status == "READY", f"lower failed: {job.status}"
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER)
D.setsockopt(zmq.HEARTBEAT_IVL, 5000); D.connect("tcp://127.0.0.1:15564")
PL = zmq.Poller(); PL.register(D, zmq.POLLIN)
fr = P.encode_pushjob("repro::gelu_out", job.pte, job.inputs, job.eager, desc="repro", user_pos=job.user_pos)
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
        print("op `gelu.out`  ->", status, detail)
        ef, df = e.float().flatten(), d.float().flatten()
        if ef.numel() == df.numel() and ef.numel() > 0:
            diff = (ef - df).abs()
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
            w = int(diff.argmax())
            print(f"  worst element [idx {w}]:  eager={ef[w].item():.6g}  device={df[w].item():.6g}")
        print("  eager  head:", ef[:6].tolist())
        print("  device head:", df[:6].tolist())
        break
