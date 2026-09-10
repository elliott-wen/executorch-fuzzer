#!/usr/bin/env python3
"""Repro: Vulkan operator bug `relu` on Moto G54 5G (Android 14).
Single-op graph fed the op's EXACT runtime inputs (baked from corpus job w102:120 out[1]),
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

SRC = 'import torch\ndef _first(r):\n    if isinstance(r, torch.Tensor): return r\n    if isinstance(r, (tuple, list)):\n        for x in r:\n            if isinstance(x, torch.Tensor): return x\n    return r\nB0 = torch.tensor([0.0, 0.0, 1.3061892986297607, 0.9737447500228882, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.526731014251709, 0.0, 0.07476574927568436, 0.0, -0.25711074471473694, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9180827140808105, 0.0, -1.1095085144042969, -0.7973700165748596, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.8714658617973328, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6717312335968018, 0.0, -0.7826655507087708, 0.0, -1.0166229009628296, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4735991954803467, -0.7379361391067505, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5760253667831421, -1.1012578010559082, 0.0, 0.0, 0.0, -0.11667953431606293, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.035021066665649414, -1.3335671424865723, 0.13162192702293396, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.8801531195640564, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7187373042106628, 0.6259779334068298, 0.05917279049754143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7752124667167664, 0.0, 0.0, 0.0, -0.14194601774215698, 0.0, 0.0, 0.8571664690971375, 0.416593998670578, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5821400880813599, -0.20241636037826538, 0.0, 0.0, 0.0, 1.3223376274108887, -0.5340818166732788, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.9165939092636108, -0.34356439113616943, 0.0, 0.0, 0.0, 0.0, 0.18678264319896698, 0.03936339169740677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6675016283988953, 0.0, 0.0, -0.5359530448913574, 0.0, -1.5954208374023438, 0.0, -0.19179311394691467, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5795247554779053, 0.7380837202072144, 0.0, 0.0, 0.0, -1.1900016069412231, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0214757919311523, 0.0, -1.084282636642456, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32).reshape([4, 4, 4, 4])\nLEAVES = [B0]\ndef g(B0):\n    T = _first(_first(torch.ops.aten.relu.default(B0)))\n    return (T,)\n'

job = build_job(SRC, "vulkan", False)
assert job.status == "READY", f"lower failed: {job.status}"
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER)
D.setsockopt(zmq.HEARTBEAT_IVL, 5000); D.connect("tcp://127.0.0.1:15554")
PL = zmq.Poller(); PL.register(D, zmq.POLLIN)
fr = P.encode_pushjob("repro::relu", job.pte, job.inputs, job.eager, desc="repro", user_pos=job.user_pos)
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
        print("op `relu`  ->", status, detail)
        ef, df = e.float().flatten(), d.float().flatten()
        if ef.numel() == df.numel() and ef.numel() > 0:
            diff = (ef - df).abs()
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
            w = int(diff.argmax())
            print(f"  worst element [idx {w}]:  eager={ef[w].item():.6g}  device={df[w].item():.6g}")
        print("  eager  head:", ef[:6].tolist())
        print("  device head:", df[:6].tolist())
        break
