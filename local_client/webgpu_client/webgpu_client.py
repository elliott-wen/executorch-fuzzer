"""webgpu_client.py — ExecuTorch WEBGPU EXECUTOR worker (sibling of vulkan_client.py).

Same role as the Android client / fvp_client / nxp_client / cadence_client / vgf_client: pull
`.pte` jobs from the broker, run them, return the RAW output tensors over the language-neutral
binary protocol. The feeder holds the eager reference and does the diff — this process is a
thin executor.

The ONLY thing that differs from vgf_client.py is *which delegate* runs the graph: via
`webgpu_runner.sh`, which invokes `webgpu_runner_io` — the ExecuTorch runtime linked with the
**WebGPU delegate** (webgpu_backend, on Dawn) + portable CPU kernels. Its subgraphs build compute
pipelines from SPIR-V compiled in at build time and dispatch them on a Vulkan device; here
that device is Mesa **lavapipe** (CPU software Vulkan), so they run on the host with NO GPU.
Unlike the VGF path there is no ML SDK emulation layer: this delegate uses core Vulkan
compute, not the ML tensor/graph extensions.

The delegate is NOT in any published executorch wheel — `Runtime.get().backend_registry` has
no VulkanBackend on 1.4.0.dev20260625 or on 1.4.1. It is compiled from the pytorch_ref source
by webgpu_client/build_runner.sh, which is why this client shells out to a runner binary
instead of using the wheel's pybindings.

EXPECT SKIPS. lavapipe, like SwiftShader, lacks integer dot product and 8-bit integer support
— upstream's own test_vulkan_delegate.py marks those cases @disable_test (the same
gaps apply: WebGPU consumes the identical Vulkan blob). They arrive here as
load/dispatch failures, which the seam maps to SKIP rather than to a numeric divergence.

Usage:
    python webgpu_client/webgpu_client.py --host <broker-ip>
    python webgpu_client/webgpu_client.py --selftest    # plumbing test, no runner/Vulkan needed
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU-only

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import zmq

HERE = Path(__file__).resolve().parent
IMPORT_ROOT = HERE.parent.parent.parent                      # dir on sys.path so `import mobile` resolves
sys.path.insert(0, str(IMPORT_ROOT))

from mobile.executor import protocol as P                  # noqa: E402
import torch  # noqa: E402

WEBGPU_RUNNER = HERE / "webgpu_runner.sh"


def _rebuild_outputs(work: Path, out_metas: list, user_pos: list | None) -> list:
    """Rebuild the runner's output tensors. webgpu_runner.sh wrote each .pte output as raw
    little-endian bytes to out_<i>.bin; dtype/shape come from the job's carried out_metas
    (USER outputs, in order) + user_pos (their .pte indices). Identical to
    cadence_client/fvp_client._rebuild_outputs."""
    files = sorted(work.glob("out_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    n = len(files)
    pos = user_pos if user_pos is not None else list(range(len(out_metas)))
    outs = [torch.empty(0)] * max(n, (max(pos) + 1 if pos else 0))
    for k, p in enumerate(pos):
        if k < len(out_metas):
            outs[p] = P.tensor_from_meta_blob(out_metas[k], (work / f"out_{p}.bin").read_bytes())
    return outs


class VulkanExecutor:
    """Runs one `.pte` via the Vulkan delegate on lavapipe and returns its outputs. Always
    returns RESULT frames: RAN with outputs, or SKIP/CRASH/TIMEOUT with a detail string."""

    def __init__(self, target: str, runner_bin: str | None, selftest: bool = False):
        self.target = target
        self.runner_bin = runner_bin
        self.selftest = selftest

    def run(self, job_id: str, pte: bytes, inputs: list, timeout: float,
            out_metas: list | None = None, user_pos: list | None = None) -> list:
        if self.selftest:
            return P.encode_result(job_id, "RAN", "selftest-echo", [t.clone() for t in inputs])

        with tempfile.TemporaryDirectory(prefix="webgpu_job_") as d:
            work = Path(d)
            pte_path = work / "model.pte"
            pte_path.write_bytes(pte)
            cmd = [str(WEBGPU_RUNNER), "--pte", str(pte_path), "--out", str(work),
                   "--target", self.target]
            for i, t in enumerate(inputs):
                _, raw = P.tensor_to_meta_blob(t.contiguous())
                ip = work / f"in_{i}.bin"
                ip.write_bytes(raw)
                cmd += ["--input", str(ip)]
            if self.runner_bin:
                cmd += ["--runner", self.runner_bin]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return P.encode_result(job_id, "TIMEOUT", "Vulkan runner exceeded job-timeout", None)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-300:]
                # exit 2 = wrapper's "not runnable on VKML" (unsupported/load/dispatch) -> SKIP;
                # exit 3 = env not set up (no runner / no emulation layer) -> also SKIP (not a
                # numeric divergence); anything else -> CRASH.
                status = "SKIP" if proc.returncode in (2, 3) else "CRASH"
                return P.encode_result(job_id, status, f"webgpu_runner rc={proc.returncode}: {tail}", None)

            try:
                outs = _rebuild_outputs(work, out_metas or [], user_pos)
            except Exception as e:
                return P.encode_result(job_id, "SKIP", f"output decode {type(e).__name__}: {e}", None)
            return P.encode_result(job_id, "RAN", "", outs)

    def close(self):
        pass


def run_client(host: str, client_port: int, ctrl_port: int, executor: VulkanExecutor,
               label: str, job_timeout: float) -> int:
    """LRU work-pull from the broker, identical handshake to cadence_client/net.client."""
    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.HEARTBEAT_IVL, 5000)
    req.setsockopt(zmq.HEARTBEAT_TIMEOUT, 20000)
    req.setsockopt(zmq.HEARTBEAT_TTL, 20000)
    req.connect(f"tcp://{host}:{client_port}")
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{host}:{ctrl_port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    poller = zmq.Poller()
    poller.register(req, zmq.POLLIN)
    poller.register(sub, zmq.POLLIN)

    print(f"  webgpu_client[{label}] → broker tcp://{host}:{client_port} (target={executor.target})",
          flush=True)
    req.send_multipart([b"READY"])
    ran = 0
    try:
        while True:
            socks = dict(poller.poll(timeout=1000))
            if sub in socks and sub.recv() == b"STOP":
                break
            if req not in socks:
                continue
            frames = req.recv_multipart()
            try:
                job_id, pte, inputs = P.decode_job(frames)
                out_metas, user_pos = P.job_output_info(frames)
                result = executor.run(job_id, pte, inputs, job_timeout, out_metas, user_pos)
            except Exception as e:
                result = P.encode_result("?", "SKIP", f"client decode {type(e).__name__}: {e}", None)
            req.send_multipart([b"RESULT", *result])
            ran += 1
            if ran % 50 == 0:
                print(f"  webgpu_client[{label}] ran {ran} jobs", flush=True)
    finally:
        executor.close()
        req.close(linger=0)
        sub.close(linger=0)
    print(f"  webgpu_client[{label}] done — ran {ran} jobs", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ExecuTorch Vulkan host (lavapipe CPU Vulkan) executor client")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--client-port", type=int, default=15555)
    ap.add_argument("--ctrl-port", type=int, default=15556)
    ap.add_argument("--label", default="webgpu")
    ap.add_argument("--job-timeout", type=float, default=120.0,
                    help="per-job wall-clock budget (VKML JIT+CPU-Vulkan is slow); TIMEOUT past this")
    ap.add_argument("--target", default="vkml", help="host target (vkml is the only one)")
    ap.add_argument("--runner", default=None,
                    help="prebuilt webgpu_runner_io binary (else webgpu_runner.sh default)")
    ap.add_argument("--selftest", action="store_true",
                    help="echo inputs as outputs (exercise the broker path without the runner)")
    a = ap.parse_args()
    ex = VulkanExecutor(a.target, a.runner, selftest=a.selftest)
    return run_client(a.host, a.client_port, a.ctrl_port, ex, a.label, a.job_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
