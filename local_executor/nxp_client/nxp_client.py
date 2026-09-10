"""nxp_client.py — NXP Neutron EXECUTOR worker (the eIQ NSYS analog of fvp_client.py).

Same role as the Android client / fvp_client: pull `.pte` jobs from the broker, run them,
return the RAW output tensors over the language-neutral binary protocol. The feeder holds
the eager (quantized) reference and does the diff — this process is a thin executor.

The ONLY thing that differs from fvp_client.py is *where* the `.pte` runs: on the **eIQ
NSYS host simulator** (a bit-exact Neutron NPU C-model) via `nxp_runner.sh`, instead of the
Corstone FVP. NXP is int8 (`quant=ALWAYS`): the `.pte`'s graph outputs are dequantized to
float at the partition boundary, so the tensors returned are float32 and diff against the
stored QUANTIZED reference (a feeder/compare concern, not this client's).

Usage:
    python nxp_client/nxp_client.py --host <broker-ip>
    python nxp_client/nxp_client.py --selftest          # plumbing test, no runner/sim needed
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

from mobile.net import protocol as P                  # noqa: E402
import torch  # noqa: E402

NXP_RUNNER = HERE / "nxp_runner.sh"


def _rebuild_outputs(work: Path, out_metas: list, user_pos: list | None) -> list:
    """Rebuild the sim's output tensors. nxp_runner.sh wrote each .pte output as raw
    little-endian bytes to out_<i>.bin. We can't introspect an NXP .pte on THIS host (the
    Neutron delegate isn't registered in the Python runtime), so dtype/shape come from the
    job's carried out_metas (USER outputs, in order) + user_pos (their .pte indices).
    Identical to fvp_client._rebuild_outputs."""
    files = sorted(work.glob("out_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    n = len(files)
    pos = user_pos if user_pos is not None else list(range(len(out_metas)))
    outs = [torch.empty(0)] * max(n, (max(pos) + 1 if pos else 0))
    for k, p in enumerate(pos):
        if k < len(out_metas):
            outs[p] = P.tensor_from_meta_blob(out_metas[k], (work / f"out_{p}.bin").read_bytes())
    return outs


class NxpExecutor:
    """Runs one `.pte` on the eIQ NSYS simulator and returns its outputs. Always returns
    RESULT frames: RAN with outputs, or SKIP/CRASH/TIMEOUT with a detail string."""

    def __init__(self, target: str, runner_bin: str | None, selftest: bool = False):
        self.target = target
        self.runner_bin = runner_bin
        self.selftest = selftest

    def run(self, job_id: str, pte: bytes, inputs: list, timeout: float,
            out_metas: list | None = None, user_pos: list | None = None) -> list:
        if self.selftest:
            return P.encode_result(job_id, "RAN", "selftest-echo", [t.clone() for t in inputs])

        with tempfile.TemporaryDirectory(prefix="nxp_job_") as d:
            work = Path(d)
            pte_path = work / "model.pte"
            pte_path.write_bytes(pte)
            cmd = [str(NXP_RUNNER), "--pte", str(pte_path), "--out", str(work),
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
                return P.encode_result(job_id, "TIMEOUT", "nsys sim exceeded job-timeout", None)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-300:]
                # exit 2 = wrapper's "not runnable on the sim" -> SKIP; else hard failure -> CRASH.
                status = "SKIP" if proc.returncode == 2 else "CRASH"
                return P.encode_result(job_id, status, f"nxp_runner rc={proc.returncode}: {tail}", None)

            try:
                outs = _rebuild_outputs(work, out_metas or [], user_pos)
            except Exception as e:
                return P.encode_result(job_id, "SKIP", f"output decode {type(e).__name__}: {e}", None)
            return P.encode_result(job_id, "RAN", "", outs)

    def close(self):
        pass


def run_client(host: str, client_port: int, ctrl_port: int, executor: NxpExecutor,
               label: str, job_timeout: float) -> int:
    """LRU work-pull from the broker, identical handshake to fvp_client/net.client."""
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

    print(f"  nxp_client[{label}] → broker tcp://{host}:{client_port} (target={executor.target})",
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
                print(f"  nxp_client[{label}] ran {ran} jobs", flush=True)
    finally:
        executor.close()
        req.close(linger=0)
        sub.close(linger=0)
    print(f"  nxp_client[{label}] done — ran {ran} jobs", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="NXP Neutron NSYS-simulator executor client")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--client-port", type=int, default=15555)
    ap.add_argument("--ctrl-port", type=int, default=15556)
    ap.add_argument("--label", default="nxp")
    ap.add_argument("--job-timeout", type=float, default=120.0,
                    help="per-job wall-clock budget (the sim is slow); TIMEOUT past this")
    ap.add_argument("--target", default="imxrt700", help="Neutron target (SDK target dir name)")
    ap.add_argument("--runner", default=None,
                    help="prebuilt nxp_executor_runner binary (else nxp_runner.sh default)")
    ap.add_argument("--selftest", action="store_true",
                    help="echo inputs as outputs (exercise the broker path without the sim)")
    a = ap.parse_args()
    ex = NxpExecutor(a.target, a.runner, selftest=a.selftest)
    return run_client(a.host, a.client_port, a.ctrl_port, ex, a.label, a.job_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
