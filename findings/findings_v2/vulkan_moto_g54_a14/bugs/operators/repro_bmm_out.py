#!/usr/bin/env python3
"""Repro: Vulkan operator bug `bmm.out` on Moto G54 5G (Android 14).
Single-op graph fed the op's EXACT runtime inputs (baked from corpus job w105:184 out[1]),
lowered to Vulkan, run on the connected device. Prints eager vs device at the worst element.
Needs a connected Vulkan worker (broker 127.0.0.1:15554).
"""
import os, sys, time, warnings, logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, "/data/jwen929")
import torch, zmq
from mobile.gen.export import build_job
from mobile.executor import protocol as P, compare as cmp

SRC = 'import torch\ndef _first(r):\n    if isinstance(r, torch.Tensor): return r\n    if isinstance(r, (tuple, list)):\n        for x in r:\n            if isinstance(x, torch.Tensor): return x\n    return r\nB0 = torch.tensor([-0.3014317452907562, 1.2009896039962769, -1.0800402164459229, 0.009261418133974075, 0.6717488765716553, -0.7337649464607239, -0.31922459602355957, 2.0304713249206543, -0.04618731513619423, 0.8646032810211182, -0.40151748061180115, 0.6710871458053589, 0.43770769238471985, -0.6160026788711548, -0.12478721141815186, -0.7518788576126099, -1.2597390413284302, -0.5189229249954224, 0.17354124784469604, -0.08023745566606522, -0.06690700352191925, 2.1082935333251953, 0.21067741513252258, 0.7898312211036682, 1.9529149532318115, -2.56963849067688, 0.041721686720848083, -1.1364902257919312, 1.2398172616958618, 1.0810905694961548, 0.13068710267543793, 1.309125542640686, -0.24038664996623993, -0.723416268825531, 1.2662569284439087, 1.5160020589828491, -0.05742755904793739, 2.0073604583740234, -0.2182280719280243, 0.7867663502693176, -0.6120952367782593, -1.7943291664123535, 1.005920648574829, -0.18970313668251038, -0.040814053267240524, 0.9263361096382141, -1.103294014930725, -0.8369443416595459], dtype=torch.float32).reshape([4, 3, 4])\nB1 = torch.tensor([1.177188515663147, 0.9062798619270325, -0.7778245806694031, -1.2655949592590332, 0.16878916323184967, 0.12994539737701416, -0.11152704805135727, -0.18146517872810364, -1.3279019594192505, -1.022309422492981, 0.8774081468582153, 1.4276269674301147, -0.9899312257766724, -0.7621164917945862, 0.6540947556495667, 1.064274787902832, 0.24195165932178497, -0.13889580965042114, -0.15280987322330475, -0.1509552299976349, 0.09610830992460251, -0.05517234653234482, -0.06069931015372276, -0.05996260046958923, 3.248009204864502, -1.8645660877227783, -2.051351547241211, -2.026454210281372, -0.3305053412914276, 0.18973131477832794, 0.20873789489269257, 0.20620444416999817, -0.3880566656589508, 0.9168955683708191, -1.0110456943511963, -0.10712146013975143, 0.6018000245094299, -1.4219257831573486, 1.5679343938827515, 0.16612444818019867, 0.33703362941741943, -0.7963389754295349, 0.8781099319458008, 0.09303676337003708, -0.13283121585845947, 0.3138519823551178, -0.34607943892478943, -0.036667514592409134, -0.03939244523644447, 0.029223455116152763, -0.0028843297623097897, 0.03862500563263893, 1.7337620258331299, -1.2861987352371216, 0.12694670259952545, -1.6999849081039429, 0.05177688971161842, -0.03841090574860573, 0.0037911233957856894, -0.0507681742310524, -1.6183356046676636, 1.2005691528320312, -0.11849513649940491, 1.586807131767273], dtype=torch.float32).reshape([4, 4, 4])\nLEAVES = [B0, B1]\ndef g(B0, B1):\n    T = _first(_first(torch.ops.aten.bmm.out(B0, B1, out=torch.empty((4, 3, 4), dtype=torch.float32))))\n    return (T,)\n'

job = build_job(SRC, "vulkan", False)
assert job.status == "READY", f"lower failed: {job.status}"
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER)
D.setsockopt(zmq.HEARTBEAT_IVL, 5000); D.connect("tcp://127.0.0.1:15554")
PL = zmq.Poller(); PL.register(D, zmq.POLLIN)
fr = P.encode_pushjob("repro::bmm_out", job.pte, job.inputs, job.eager, desc="repro", user_pos=job.user_pos)
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
        print("op `bmm.out`  ->", status, detail)
        ef, df = e.float().flatten(), d.float().flatten()
        if ef.numel() == df.numel() and ef.numel() > 0:
            diff = (ef - df).abs()
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
            w = int(diff.argmax())
            print(f"  worst element [idx {w}]:  eager={ef[w].item():.6g}  device={df[w].item():.6g}")
        print("  eager  head:", ef[:6].tolist())
        print("  device head:", df[:6].tolist())
        break
