#!/usr/bin/env python3
"""repro_dropped_store.py — VGF GRAPH-OPTIMIZATION bug (mismatch / graph-optimization).

**Dropped store under the multi-output memory plan.** A VGF-delegated op computes the CORRECT
value when its output is returned ALONE, but is written **all-zeros** on the device the moment a
**view/copy-family output** (squeeze_copy / clone / alias_copy / transpose_copy / permute_copy /
view_copy / expand_copy / split_with_sizes_copy / …) is co-returned alongside it. The op's kernel
is fine — the multi-output memory planner drops the target's store (copy-elision / buffer-aliasing),
leaving its output buffer at its zero-initialized value.

This is NOT an operator bug: `bisect_output_set` shows the target is CONE_LOCAL-correct when returned
alone; it only diverges in company. Device-verified A/B below. 239 such ZEROED cases in corpus_v4/vgf
(the #1 VGF graph-opt mechanism); the target op is arbitrary (prod/fmod/max/div/min/…), the TRIGGER is
a co-returned view/copy output. Analog of the Vulkan copy-elision/aliasing bug.

Needs the AoT env (rebuilds sub-graphs): PATH=.venv/bin, PYTHONPATH=/data/jwen929,
MODEL_CONVERTER_LIB_DIR=<libstdc++ w/ GLIBCXX_3.4.30>. Broker + a vgf_client worker must be up.

Run:
  PATH=/data/jwen929/mobile/.venv/bin:$PATH PYTHONPATH=/data/jwen929 \
  MODEL_CONVERTER_LIB_DIR=$(cat /data/jwen929/mobile/tmp/vgf_libdir.txt) BROKER_PORT=15774 \
  /data/jwen929/mobile/.venv/bin/python findings/vgf/bugs/repro_dropped_store.py
"""
import os, sys, time
sys.path.insert(0, "/data/jwen929")
import torch, zmq
from mobile.gen.export import build_job
from mobile.net import protocol as P, compare as cmp, corpus as C
from mobile.gen.diff.cone_predicate import output_set_localizer

CO = "/data/jwen929/mobile/corpus_v4/vgf"
PORT = int(os.environ.get("BROKER_PORT", "15774"))
ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER); D.connect(f"tcp://127.0.0.1:{PORT}")
POL = zmq.Poller(); POL.register(D, zmq.POLLIN); seq = [0]

# (job, out_idx, target_op, trigger_view_op) — device-verified ZEROED dropped-store cases
CASES = [("w108:171", 0, "fmod", "split_with_sizes_copy"),
         ("w111:578", 0, "min",  "log10 (+ view siblings)"),
         ("w11:274",  2, "div",  "logical_xor (+ view siblings)"),
         ("w10:13",   1, "squeeze_copy", "constant_pad_nd")]

def run(src):
    j = build_job(src, "vgf", False)
    if j.status != "READY":
        return None, None, j.status
    seq[0] += 1
    fr = P.encode_pushjob(f"ds::{seq[0]}", j.pte, j.inputs, j.eager, desc="ds", user_pos=j.user_pos)
    D.send_multipart(P.job_frames_from_pushjob(fr)); t = time.monotonic()
    while time.monotonic() - t < 60:
        if dict(POL.poll(1000)).get(D):
            _, st, det, raw = P.decode_result(D.recv_multipart())
            et = cmp.select(raw, j.user_pos)
            if len(et) > len(j.eager): et = et[len(et) - len(j.eager):]
            return j.eager, et, st
    return None, None, "TIMEOUT"

print("VGF dropped-store (ZEROED) graph-opt bug — device A/B\n" + "=" * 64)
for job, oi, top, trig in CASES:
    py = C.job_file(CO, job, "py")
    others, bos, target, ret = output_set_localizer(py, oi)
    eaA, etA, stA = run(bos([target]))                 # A: target returned ALONE
    eaB, etB, stB = run(bos(ret))                      # B: target + its real siblings
    pos = [o for o in ret if o in set(ret)].index(target)
    okA = etA is not None and cmp._cmp(eaA[0], etA[0], 0)[0] == "OK"
    devB = etB[pos] if etB is not None else torch.empty(0)
    zeroed = devB.numel() and float(devB.abs().max()) == 0.0 and float(eaB[pos].abs().max()) > 0
    print(f"\n{job} out[{oi}]  target={top}  trigger sibling={trig}")
    print(f"  A  target ALONE      : device={etA[0].flatten()[:3].tolist() if etA else stA}  -> {'CORRECT' if okA else 'MISMATCH'}")
    print(f"  B  target + siblings : eager={eaB[pos].flatten()[:3].tolist()}  device={devB.flatten()[:3].tolist()}")
    print(f"     => {'BUG: store DROPPED (device all-zeros, kernel was correct alone)' if zeroed else 'not reproduced'}")
