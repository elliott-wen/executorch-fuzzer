#!/usr/bin/env python3
"""Repro: Vulkan operator bug `argmin.out` on Moto G54 5G (Android 14).
Single-op graph fed the op's EXACT runtime inputs (baked from corpus job w103:290 out[1]),
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

SRC = 'import torch\ndef _first(r):\n    if isinstance(r, torch.Tensor): return r\n    if isinstance(r, (tuple, list)):\n        for x in r:\n            if isinstance(x, torch.Tensor): return x\n    return r\nB0 = torch.tensor([-1.157940149307251, -0.16602927446365356, 1.3061892986297607, 0.9737447500228882, 0.1882219910621643, 0.07476574927568436, 2.526731014251709, -0.25711074471473694, 0.9180827140808105, -1.423766851425171, -0.7973700165748596, 0.314258337020874, -0.021913748234510422, 0.964479923248291, 0.028803128749132156, -0.9002689719200134, -1.0166229009628296, -0.7826655507087708, 0.6717312335968018, 1.0929709672927856, 1.285459041595459, -0.7379361391067505, -0.8118598461151123, -0.8020063042640686, -0.422681599855423, 0.9987069368362427, -1.1012578010559082, -0.11667953431606293, 1.7976133823394775, -1.3335671424865723, 0.13162192702293396, -1.7625923156738281, 0.815504789352417, 0.8116468191146851, -1.1732007265090942, -0.5185992121696472, 0.6259779334068298, 0.7187373042106628, 0.05917279049754143, -1.5621888637542725, -0.7752124667167664, 0.416593998670578, 0.8571664690971375, -0.14194601774215698, 1.3223376274108887, 1.5821400880813599, -0.5340818166732788, -0.20241636037826538, -1.9165939092636108, 0.18678264319896698, 0.03936339169740677, -0.34356439113616943, -0.19179311394691467, -1.5954208374023438, -0.5359530448913574, -0.6675016283988953, 0.5795247554779053, -1.1900016069412231, -0.1895504742860794, 0.9276341795921326, 1.1686367988586426, -1.084282636642456, -1.0214757919311523, 0.9017773866653442, -1.2836817502975464, -0.9814533591270447, 1.1956896781921387, -0.9490326046943665, 0.681002676486969, -0.25170546770095825, -0.6112949252128601, 0.7316865921020508, -0.41853776574134827, 0.18495789170265198, 0.38769394159317017, -0.15291588008403778, 0.7195093035697937, -2.033902645111084, 1.0077474117279053, -0.30521777272224426, -0.3014317452907562, 1.2009896039962769, -1.0800402164459229, 0.009261418133974075, 0.6717488765716553, -0.7337649464607239, -0.31922459602355957, 2.0304713249206543, -0.04618731513619423, 0.8646032810211182, -0.40151748061180115, 0.6710871458053589, 0.43770769238471985, -0.6160026788711548, -0.12478721141815186, -0.7518788576126099, -1.2597390413284302, -0.5189229249954224, 0.17354124784469604, -0.08023745566606522, -0.06690700352191925, 2.1082935333251953, 0.21067741513252258, 0.7898312211036682, 1.9529149532318115, -2.56963849067688, 0.041721686720848083, -1.1364902257919312, 1.2398172616958618, 1.0810905694961548, 0.13068710267543793, 1.309125542640686, -0.24038664996623993, -0.723416268825531, 1.2662569284439087, 1.5160020589828491, -0.05742755904793739, 2.0073604583740234, -0.2182280719280243, 0.7867663502693176, -0.6120952367782593, -1.7943291664123535, 1.005920648574829, -0.18970313668251038, -0.040814053267240524, 0.9263361096382141, -1.103294014930725, -0.8369443416595459], dtype=torch.float32).reshape([4, 4, 4, 2])\nLEAVES = [B0]\ndef g(B0):\n    T = _first(_first(torch.ops.aten.argmin.out(B0, None, False, out=torch.empty((), dtype=torch.int64))))\n    return (T,)\n'

job = build_job(SRC, "vulkan", False)
assert job.status == "READY", f"lower failed: {job.status}"
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER)
D.setsockopt(zmq.HEARTBEAT_IVL, 5000); D.connect("tcp://127.0.0.1:15554")
PL = zmq.Poller(); PL.register(D, zmq.POLLIN)
fr = P.encode_pushjob("repro::argmin_out", job.pte, job.inputs, job.eager, desc="repro", user_pos=job.user_pos)
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
        print("op `argmin.out`  ->", status, detail)
        ef, df = e.float().flatten(), d.float().flatten()
        if ef.numel() == df.numel() and ef.numel() > 0:
            diff = (ef - df).abs()
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
            w = int(diff.argmax())
            print(f"  worst element [idx {w}]:  eager={ef[w].item():.6g}  device={df[w].item():.6g}")
        print("  eager  head:", ef[:6].tolist())
        print("  device head:", df[:6].tolist())
        break
