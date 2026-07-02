#!/usr/bin/env python3
"""Repro: Vulkan operator bug `lift_fresh_copy` on Moto G54 5G (Android 14).
Single-op graph fed the op's EXACT runtime inputs (baked from corpus job w100:65 out[0]),
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

SRC = 'import torch\ndef _first(r):\n    if isinstance(r, torch.Tensor): return r\n    if isinstance(r, (tuple, list)):\n        for x in r:\n            if isinstance(x, torch.Tensor): return x\n    return r\nB0 = torch.tensor([1.177188515663147, 0.9062798619270325, -0.7778245806694031, -1.2655949592590332, 0.16878916323184967, 0.12994539737701416, -0.11152704805135727, -0.18146517872810364, -1.3279019594192505, -1.022309422492981, 0.8774081468582153, 1.4276269674301147, -0.9899312257766724, -0.7621164917945862, 0.6540947556495667, 1.064274787902832, 0.24195165932178497, -0.13889580965042114, -0.15280987322330475, -0.1509552299976349, 0.09610830992460251, -0.05517234653234482, -0.06069931015372276, -0.05996260046958923, 3.248009204864502, -1.8645660877227783, -2.051351547241211, -2.026454210281372, -0.3305053412914276, 0.18973131477832794, 0.20873789489269257, 0.20620444416999817, -0.3880566656589508, 0.9168955683708191, -1.0110456943511963, -0.10712146013975143, 0.6018000245094299, -1.4219257831573486, 1.5679343938827515, 0.16612444818019867, 0.33703362941741943, -0.7963389754295349, 0.8781099319458008, 0.09303676337003708, -0.13283121585845947, 0.3138519823551178, -0.34607943892478943, -0.036667514592409134, -0.03939244523644447, 0.029223455116152763, -0.0028843297623097897, 0.03862500563263893, 1.7337620258331299, -1.2861987352371216, 0.12694670259952545, -1.6999849081039429, 0.05177688971161842, -0.03841090574860573, 0.0037911233957856894, -0.0507681742310524, -1.6183356046676636, 1.2005691528320312, -0.11849513649940491, 1.586807131767273], dtype=torch.float32).reshape([4, 4, 4])\nLEAVES = [B0]\ndef g(B0):\n    T = _first(_first(torch.ops.aten.lift_fresh_copy.default(B0)))\n    return (T,)\n'

job = build_job(SRC, "vulkan", False)
assert job.status == "READY", f"lower failed: {job.status}"
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER)
D.setsockopt(zmq.HEARTBEAT_IVL, 5000); D.connect("tcp://127.0.0.1:15554")
PL = zmq.Poller(); PL.register(D, zmq.POLLIN)
fr = P.encode_pushjob("repro::lift_fresh_copy", job.pte, job.inputs, job.eager, desc="repro", user_pos=job.user_pos)
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
        print("op `lift_fresh_copy`  ->", status, detail)
        ef, df = e.float().flatten(), d.float().flatten()
        if ef.numel() == df.numel() and ef.numel() > 0:
            diff = (ef - df).abs()
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
            w = int(diff.argmax())
            print(f"  worst element [idx {w}]:  eager={ef[w].item():.6g}  device={df[w].item():.6g}")
        print("  eager  head:", ef[:6].tolist())
        print("  device head:", df[:6].tolist())
        break
