#!/usr/bin/env python3
"""repro_var_correction.py — MINIMAL repro: CoreML var.correction ignores correction=None default.

torch.var(x, correction=None) means the UNBIASED variance (÷(N-1)); CoreML computes the BIASED
one (÷N, i.e. correction=0), so the result is off by exactly (N-1)/N. Correct on portable/xnnpack.
This is an OPERATOR bug — it surfaced in the corpus as a "SCALE:(N-1)/N" on downstream consumers
of var (div/mul/select/…), which the bisector labeled COMPOSITIONAL because baking var's correct
value hides it. Root cause is var itself.

  BROKER_PORT=15554 PYTHONPATH=/data/jwen929 /data/jwen929/mobile/.venv/bin/python repro_var_correction.py
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import zmq
from mobile.gen.export import build_job
from mobile.executor import protocol as P, compare as cmp

def make_src(N, corr):
    return (f"import torch\ntorch.manual_seed(1)\nL0=torch.randn({N})\n"
            f"def g(L0):\n    return (torch.ops.aten.var.correction(L0, None, correction={corr}),)\nLEAVES=[L0]\n")

def run(src, backend):
    j = build_job(src, backend)
    if j.status != "READY": return None, None, f"BUILD:{j.status}"
    if backend != "coreml":
        from mobile.executor.et_runner import run_pte
        return j.eager[-1].flatten()[0].item(), run_pte(j.pte, j.inputs)[-1].flatten()[0].item(), "RAN"
    ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER); D.connect(f"tcp://127.0.0.1:{os.environ.get('BROKER_PORT','15554')}")
    POL = zmq.Poller(); POL.register(D, zmq.POLLIN)
    fr = P.encode_pushjob("vc::1", j.pte, j.inputs, j.eager, desc="vc", user_pos=j.user_pos)
    D.send_multipart(P.job_frames_from_pushjob(fr)); t0 = time.monotonic()
    while time.monotonic() - t0 < 90:
        if dict(POL.poll(1000)).get(D):
            _, st, det, raw = P.decode_result(D.recv_multipart())
            if st != "RAN": return None, None, st
            et = cmp.select(raw, j.user_pos)
            if len(et) > len(j.eager): et = et[len(et)-len(j.eager):]
            return j.eager[-1].flatten()[0].item(), et[-1].flatten()[0].item(), "RAN"
    return None, None, "TIMEOUT"

if __name__ == "__main__":
    for N in (2, 4):
        e, dp, _ = run(make_src(N, "None"), "portable")
        e2, dc, st = run(make_src(N, "None"), "coreml")
        print(f"var.correction(randn({N}), correction=None):  eager(unbiased)={e:.5f}  "
              f"portable={dp:.5f}  coreml={dc:.5f}  coreml/eager={dc/e:.4f}  [(N-1)/N={(N-1)/N:.3f}]")
    print("=> CoreML returns the BIASED variance (÷N) for correction=None; PyTorch/portable/xnnpack "
          "return UNBIASED (÷(N-1)). Operator bug, off by (N-1)/N.")
