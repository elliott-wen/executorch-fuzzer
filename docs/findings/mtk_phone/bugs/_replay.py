"""Shared replay helper for mtk_phone repros.

Each repro replays the EXACT one-op corpus .job (pte + embedded eager reference + inputs) on the
connected MediaTek phone worker via the broker, and prints eager-vs-device. It does NOT rebuild the
.pte — the corpus .job is self-contained (mtk lowering needs the cp310 mtk_converter; the phone
executor only needs the pte bytes), so this runs in the plain 3.12 .venv with just the broker up.

Broker (a human starts it): job-port 15554 (feeders), client-port 15555 (phone), ctrl 15556.
Usage:  python repro_<op>.py            # replays the canonical job_id(s) baked into that repro
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import zmq
from mobile.executor import corpus as C
from mobile.executor import protocol as P
from mobile.executor import compare as cmp

CORPUS = "/data/jwen929/mobile/corpus_v3/mtk"
HOST = os.environ.get("BROKER_HOST", "127.0.0.1")
JOB_PORT = int(os.environ.get("BROKER_JOB_PORT", "15554"))


def replay(job_id, n=5, timeout=40):
    """Feed one corpus job_id n times; return list of (status, detail, eager, et)."""
    path = C.job_file(CORPUS, job_id)
    ctx = zmq.Context.instance()
    out = []
    for _ in range(n):
        d = ctx.socket(zmq.DEALER); d.setsockopt(zmq.HEARTBEAT_IVL, 5000)
        d.connect(f"tcp://{HOST}:{JOB_PORT}")
        pl = zmq.Poller(); pl.register(d, zmq.POLLIN)
        frames = C.read_job(path)
        jid, desc = P.peek_jobinfo(frames)
        eager, upos = P.eager_from_pushjob(frames)
        d.send_multipart(P.job_frames_from_pushjob(frames))
        t0 = time.monotonic(); res = ("TIMEOUT", "no result", None, None)
        while time.monotonic() - t0 < timeout:
            if dict(pl.poll(1000)).get(d):
                _jid, st, det, raw = P.decode_result(d.recv_multipart())
                if st == "RAN":
                    et = cmp.select(raw, upos)
                    if len(et) > len(eager): et = et[len(et)-len(eager):]
                    s, dd = cmp.compare(eager, et)
                    res = (s, dd, eager, et)
                else:
                    res = (st, det, eager, None)
                break
        out.append(res); d.close()
    return desc, out


def show(job_id, label, n=5):
    print(f"\n=== {label}  (job {job_id}, {n}x) ===")
    desc, runs = replay(job_id, n)
    print(f"graph: {desc}")
    sts = [r[0] for r in runs]
    print(f"verdicts: {sts}   (deterministic={len(set(sts))==1})")
    s, det, eager, et = runs[0]
    print(f"first-run: {s}  {det[:120]}")
    if eager is not None:
        for i, e in enumerate(eager):
            ev = e.flatten().tolist()[:12]
            print(f"  out[{i}] dtype={e.dtype} shape={list(e.shape)}")
            print(f"    eager : {ev}")
            if et is not None and i < len(et):
                print(f"    device: {et[i].flatten().tolist()[:12]}")
    return sts
