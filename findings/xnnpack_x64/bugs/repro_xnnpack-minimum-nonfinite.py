#!/usr/bin/env python3
"""Repro: XNNPACK delegated bug `minimum` on a non-finite/boundary input (x86 host, xnnpack_client).
minimum(NaN,finite) returns the finite operand on XNNPACK (SIMD fmin launders NaN) where ATen returns NaN.
This exact trigger value is under-sampled by the corpus (uniform random boundary injection), so it is
fed directly here. Needs a connected xnnpack worker (BROKER_JOB_PORT, default 15564).
"""
import os,sys,time,warnings,logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES","");warnings.filterwarnings("ignore");logging.disable(logging.WARNING)
sys.path.insert(0,"/data/jwen929")
import torch,zmq
from mobile.gen.export import build_job
from mobile.net import protocol as P, compare as cmp
SRC="import torch\ndef _f(r):\n return r[0] if isinstance(r,(tuple,list)) else r\nL0=torch.tensor([float('nan'),2.0]);L1=torch.tensor([5.0,5.0])\ndef g(L0,L1):\n return (_f(torch.ops.aten.minimum.default(L0,L1)),)\nLEAVES=[L0,L1]\n"
PORT=int(os.environ.get("BROKER_JOB_PORT","15564"))
job=build_job(SRC,"xnnpack",False)
if job.status!="READY": print(f"build {job.status}: {job.detail[:150]}");sys.exit(0)
ctx=zmq.Context.instance();D=ctx.socket(zmq.DEALER);D.setsockopt(zmq.HEARTBEAT_IVL,5000);D.connect(f"tcp://127.0.0.1:{PORT}")
PL=zmq.Poller();PL.register(D,zmq.POLLIN)
fr=P.encode_pushjob("repro::minimum",job.pte,job.inputs,job.eager,desc="repro",user_pos=job.user_pos,delegated={"ops":1})
D.send_multipart(P.job_frames_from_pushjob(fr));t0=time.monotonic()
while time.monotonic()-t0<60:
    if dict(PL.poll(1000)).get(D):
        _,st,det,raw=P.decode_result(D.recv_multipart())
        if st!="RAN":print(f"device {st}: {det[:150]}");break
        et=cmp.select(raw,job.user_pos)
        if len(et)>len(job.eager):et=et[len(et)-len(job.eager):]
        s,d=cmp._cmp(job.eager[0],et[0],0)
        print(f"minimum -> {s} {d}");print("  eager :",job.eager[0].tolist());print("  device:",et[0].tolist());break
