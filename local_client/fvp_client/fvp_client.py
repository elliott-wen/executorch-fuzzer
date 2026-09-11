"""fvp_client.py — Ethos-U EXECUTOR worker (the FVP analog of net/client.py).

Same role as the on-device Android client and the local `mobile client`: pull `.pte`
jobs from the broker, run them, return the RAW output tensors over the language-neutral
binary protocol. The feeder still holds the eager reference and does the diff — this
process is a thin executor, exactly like the phone.

The ONLY thing that differs from net/client.py is *where* the `.pte` runs. There is no
ExecuTorch host runtime for an Ethos-U command stream, so instead of
`executorch.runtime`, each job is run on the **Corstone FVP simulator** via a small
shell seam (`fvp_runner.sh`):

    JOB(pte+inputs) ─► write pte + in_<i>.bin ─► fvp_runner.sh ─► FVP runs the .pte
                    ◄─ out_<i>.bin + outmeta.json ◄────────────────────────────────────
    ─► RESULT(raw outputs) ─► broker ─► feeder diffs vs eager

Crash/timeout isolation mirrors net/client.py: the FVP runs as a child process; a
non-zero exit → CRASH, exceeding the deadline → TIMEOUT. The worker ALWAYS returns a
verdict.

NUMERICS NOTE (important — see README.md): Ethos-U is int8. An Ethos-U `.pte`'s graph
outputs are *dequantized back to float* at the partition boundary, so the tensors this
client returns are float32 and structurally diff-able against the float eager oracle —
but they carry quantization error. The meaningful reference for a quantized backend is a
*quantized* reference, not raw fp32 eager; that is a feeder/compare.py concern, not this
client's. This client just returns what the device produced.

Usage:
    python fvp_client/fvp_client.py --host <broker-ip> --target ethos-u55-128
    python fvp_client/fvp_client.py --selftest          # plumbing test, no FVP needed
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU-only; never touch CUDA

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import zmq

HERE = Path(__file__).resolve().parent
IMPORT_ROOT = HERE.parent.parent.parent                      # dir on sys.path so `import mobile` resolves
sys.path.insert(0, str(IMPORT_ROOT))

from mobile.net import protocol as P                  # noqa: E402

FVP_RUNNER = HERE / "fvp_runner.sh"

# torch dtype string (as written in outmeta.json) → the ScalarType int code the protocol
# uses. We round-trip through protocol.tensor_from_meta_blob so the wire encoding matches
# every other client exactly.
import torch  # noqa: E402


def _rebuild_outputs(work: Path, out_metas: list, user_pos: list | None) -> list:
    """Rebuild the device's output tensors. The FVP runner wrote each .pte output as raw
    little-endian bytes to out_<i>.bin. We can't introspect a cortex-m/ethos-u .pte on THIS
    host (its device kernels aren't registered here), so the dtype/shape come from the job's
    carried `out_metas` (the USER outputs, in user order) + `user_pos` (their .pte indices).
    Returns the FULL .pte output list so the feeder's select(user_pos) picks the right ones;
    non-user (mutated-input alias) slots get an empty placeholder — the feeder drops them."""
    files = sorted(work.glob("out_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    n = len(files)
    pos = user_pos if user_pos is not None else list(range(len(out_metas)))
    outs = [torch.empty(0)] * max(n, (max(pos) + 1 if pos else 0))
    for k, p in enumerate(pos):
        if k < len(out_metas):
            outs[p] = P.tensor_from_meta_blob(out_metas[k], (work / f"out_{p}.bin").read_bytes())
    return outs


class FvpExecutor:
    """Runs one `.pte` on the FVP and returns its outputs. Always returns RESULT frames:
    RAN with outputs, or SKIP/CRASH/TIMEOUT with a detail string (no outputs)."""

    def __init__(self, backend: str, target: str, fvp_bin: str | None, runner_elf: str | None,
                 num_macs: int, selftest: bool = False):
        self.backend = backend          # ethos-u | cortex-m — selects the runner build in fvp_runner.sh
        self.target = target
        self.fvp_bin = fvp_bin
        self.runner_elf = runner_elf
        self.num_macs = num_macs
        self.selftest = selftest

    def run(self, job_id: str, pte: bytes, inputs: list, timeout: float,
            out_metas: list | None = None, user_pos: list | None = None) -> list:
        if self.selftest:
            # Plumbing test: echo inputs back as outputs so the broker/feeder path can be
            # exercised end-to-end without an installed FVP. (Will MISMATCH; that's fine.)
            return P.encode_result(job_id, "RAN", "selftest-echo", [t.clone() for t in inputs])

        with tempfile.TemporaryDirectory(prefix="fvp_job_") as d:
            work = Path(d)
            pte_path = work / "model.pte"
            pte_path.write_bytes(pte)
            cmd = [str(FVP_RUNNER), "--backend", self.backend,
                   "--pte", str(pte_path), "--out", str(work),
                   "--target", self.target, "--macs", str(self.num_macs)]
            for i, t in enumerate(inputs):
                _, raw = P.tensor_to_meta_blob(t.contiguous())
                ip = work / f"in_{i}.bin"
                ip.write_bytes(raw)
                cmd += ["--input", str(ip)]
            if self.fvp_bin:
                cmd += ["--fvp", self.fvp_bin]
            if self.runner_elf:
                cmd += ["--runner", self.runner_elf]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return P.encode_result(job_id, "TIMEOUT", "FVP exceeded job-timeout", None)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-300:]
                # exit 2 is the wrapper's reserved "graph not runnable on this target" → SKIP;
                # anything else is treated as a hard failure of the run → CRASH.
                status = "SKIP" if proc.returncode == 2 else "CRASH"
                return P.encode_result(job_id, status, f"fvp_runner rc={proc.returncode}: {tail}", None)

            try:
                outs = _rebuild_outputs(work, out_metas or [], user_pos)
            except Exception as e:
                return P.encode_result(job_id, "SKIP", f"output decode {type(e).__name__}: {e}", None)
            return P.encode_result(job_id, "RAN", "", outs)

    def close(self):
        pass


def run_client(host: str, client_port: int, ctrl_port: int, executor: FvpExecutor,
               label: str, job_timeout: float) -> int:
    """LRU work-pull from the broker, identical handshake to net/client.py."""
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

    print(f"  fvp_client[{label}] → broker tcp://{host}:{client_port} (target={executor.target})",
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
                print(f"  fvp_client[{label}] ran {ran} jobs", flush=True)
    finally:
        executor.close()
        req.close(linger=0)
        sub.close(linger=0)
    print(f"  fvp_client[{label}] done — ran {ran} jobs", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Ethos-U FVP executor client")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--client-port", type=int, default=15555)
    ap.add_argument("--ctrl-port", type=int, default=15556)
    ap.add_argument("--label", default="fvp")
    ap.add_argument("--job-timeout", type=float, default=120.0,
                    help="per-job wall-clock budget (FVP runs are slow); TIMEOUT past this")
    ap.add_argument("--backend", default="ethos-u", choices=["ethos-u", "cortex-m"],
                    help="which Arm backend's runner to build/run on the FVP")
    ap.add_argument("--target", default="ethos-u55-128",
                    help="accelerator/CPU config (e.g. ethos-u55-128, cortex-m55)")
    ap.add_argument("--macs", type=int, default=128, help="FVP ethosu.num_macs (Ethos-U only)")
    ap.add_argument("--fvp", default=None, help="FVP binary (else fvp_runner.sh default)")
    ap.add_argument("--runner", default=None,
                    help="prebuilt semihosting runner ELF / build dir (else fvp_runner.sh builds)")
    ap.add_argument("--selftest", action="store_true",
                    help="echo inputs as outputs (exercise the broker path without an FVP)")
    a = ap.parse_args()
    ex = FvpExecutor(a.backend, a.target, a.fvp, a.runner, a.macs, selftest=a.selftest)
    return run_client(a.host, a.client_port, a.ctrl_port, ex, a.label, a.job_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
