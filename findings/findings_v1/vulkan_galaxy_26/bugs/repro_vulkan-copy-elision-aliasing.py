#!/usr/bin/env python3
"""Check the user's hypothesis: a copy op (lift_fresh_copy/clone) elided to a buffer alias in a
multi-output Vulkan graph causes the memory planner to reuse the still-live source buffer,
silently corrupting a SIBLING output (out[0]).

Test: lower Bug (uses lift_fresh_copy) and Control (uses add 0.0 -> fresh buffer) to vulkan, run
BOTH on the phone via the broker, compare device out[0] vs eager out[0]. If Bug's out[0] is
wrong but Control's is correct -> hypothesis confirmed (planner/copy-elision aliasing bug).
"""
import os, sys, time, warnings, logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore"); logging.disable(logging.INFO)
sys.path.insert(0, "/data/jwen929")
import torch, zmq
from torch.export.graph_signature import OutputKind
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from mobile.net import protocol as P
from mobile.net import compare as cmp

class Bug(torch.nn.Module):
    def forward(self, x):
        n0 = torch.ops.aten.sigmoid.default(x)
        n1 = torch.ops.aten.lift_fresh_copy.default(n0)            # copy -> elided to alias of n0
        out0 = torch.ops.aten.prod.default(
                   torch.ops.aten.acos.default(n0).to(torch.float16))
        out1 = torch.ops.aten.acosh.default(n1).to(torch.float16)  # acosh of (0,1) -> NaN
        return out0, out1

class Control(torch.nn.Module):
    def forward(self, x):
        n0 = torch.ops.aten.sigmoid.default(x)
        n1 = torch.ops.aten.add.Tensor(n0, 0.0)                   # forces a real fresh buffer
        out0 = torch.ops.aten.prod.default(
                   torch.ops.aten.acos.default(n0).to(torch.float16))
        out1 = torch.ops.aten.acosh.default(n1).to(torch.float16)
        return out0, out1

# also a clone variant of the bug
class BugClone(torch.nn.Module):
    def forward(self, x):
        n0 = torch.ops.aten.sigmoid.default(x)
        n1 = torch.ops.aten.clone.default(n0)
        out0 = torch.ops.aten.prod.default(
                   torch.ops.aten.acos.default(n0).to(torch.float16))
        out1 = torch.ops.aten.acosh.default(n1).to(torch.float16)
        return out0, out1

def lower(M, x):
    ep = torch.export.export(M(), (x,))
    exe = to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False)).to_executorch()
    specs = exe.exported_program().graph_signature.output_specs
    user_pos = [i for i, s in enumerate(specs) if s.kind == OutputKind.USER_OUTPUT]
    return bytes(exe.buffer), user_pos

def run_on_phone(dealer, poller, jid, pte, x, eager, user_pos, timeout=40):
    frames = P.encode_pushjob(jid, pte, [x], list(eager), desc=jid, user_pos=user_pos)
    dealer.send_multipart(P.job_frames_from_pushjob(frames))
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if dict(poller.poll(timeout=1000)).get(dealer):
            rjid, status, detail, outputs = P.decode_result(dealer.recv_multipart())
            if rjid != jid: continue
            return status, detail, outputs
    return "TIMEOUT", "", []

def main():
    x = torch.tensor([-0.4824913740158081, 1.1103534698486328])
    ctx = zmq.Context.instance()
    dealer = ctx.socket(zmq.DEALER); dealer.set_hwm(64)
    dealer.setsockopt(zmq.HEARTBEAT_IVL, 5000); dealer.setsockopt(zmq.HEARTBEAT_TIMEOUT, 20000)
    dealer.connect("tcp://127.0.0.1:15554")
    poller = zmq.Poller(); poller.register(dealer, zmq.POLLIN)

    for name, M in [("Bug(lift_fresh_copy)", Bug), ("BugClone(clone)", BugClone),
                    ("Control(add 0.0)", Control)]:
        eager = M()(x)
        eager = [e.detach() for e in eager]
        pte, user_pos = lower(M, x)
        status, detail, outputs = run_on_phone(dealer, poller, f"chk::{name}", pte, x, eager, user_pos)
        if status != "RAN":
            print(f"{name:22s} device status={status} {detail[:80]}")
            continue
        dev = cmp.select(outputs, user_pos)
        def fmt(t): return str([round(v, 4) for v in t.flatten().tolist()])
        e0v, d0v = eager[0].flatten().float(), dev[0].flatten().float()
        same0 = torch.allclose(e0v, d0v, atol=1e-2, rtol=1e-2, equal_nan=True)
        flag = "OK" if same0 else "  <-- out[0] CORRUPTED"
        print(f"{name:22s} out0 eager={fmt(eager[0])} device={fmt(dev[0])} {flag}")
        print(f"{'':22s} out1 eager={fmt(eager[1])} device={fmt(dev[1])}")

if __name__ == "__main__":
    main()
