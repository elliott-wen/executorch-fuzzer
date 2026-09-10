"""cuda_client.py — CUDA (AOTInductor) EXECUTOR worker (a host-runtime client like xnnpack/openvino).

Same role as the openvino/xnnpack clients: pull `.pte` jobs from the broker, run them, return the
RAW output tensors over the language-neutral binary protocol. The feeder owns the eager reference
and does the diff — this process is a thin executor.

What's special vs the other host clients: the `.pte` was lowered to the **CUDA** backend, whose
delegate blob is an AOTInductor-compiled CUDA .so. The ExecuTorch runtime's `aoti_cuda_backend`
(built into this env's from-source executorch, EXECUTORCH_BUILD_CUDA=ON) loads that .so and runs it
on an **NVIDIA GPU**. So, unlike every other client, this one must NOT hide the GPU — it deliberately
does NOT set CUDA_VISIBLE_DEVICES="" (the rest of the harness does). It runs in `.venv-cuda`
(torch+cu130), not the CPU `.venv`. See mobile/gen/export/backends/cuda.py.

Device: with N GPUs visible the delegate targets cuda:0 == the first VISIBLE device, so pin a single
GPU with `--gpu <id>` (sets CUDA_VISIBLE_DEVICES for this process) to spread a client fleet across
the box. Inputs are CPU tensors; the AOTI delegate moves them H2D/D2H, so the protocol is unchanged.

Crash isolation is identical to openvino_client: a stable COORDINATOR drives a disposable warm
EXECUTOR child. A native abort (bad kernel) kills only the child → CRASH + respawn; a hung kernel is
killed at the deadline → TIMEOUT + respawn. So the worker ALWAYS returns a verdict.

    JOB(pte+inputs) ─► warm fork runs forward(*inputs) on the ET runtime (CUDA delegate, on GPU)
                    ─► RESULT(raw outputs) ─► broker ─► feeder diffs vs eager

Usage:
    python cuda_client/cuda_client.py --host <broker-ip> [--gpu 0] [--job-timeout 60]
"""

from __future__ import annotations

import os
# NOTE: intentionally NO `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")` here — this is the one
# client that needs the GPU. --gpu (below) pins a specific device before torch is imported.

import argparse
import multiprocessing as mp
import queue as _queue
import sys
import tempfile
import time
from pathlib import Path

import zmq

HERE = Path(__file__).resolve().parent
IMPORT_ROOT = HERE.parent.parent.parent                      # dir on sys.path so `import mobile` resolves
sys.path.insert(0, str(IMPORT_ROOT))

from mobile.net import protocol as P                  # noqa: E402

BACKEND = "cuda"

# Known-NOISE substrings dropped from a captured native log (SKIP path). The CUDA backend/runtime
# is chatty: the AOTInductor compile chatter is a lower-time (pregen) thing, but the RUNTIME still
# prints per-execution lines from cuda_backend.cpp. None of these are failures on their own — drop
# them, keep everything else (a real native abort message survives).
_NOISE = ("cuda_backend.cpp", "Created new CUDA stream", "container_handle", "weights_blob",
          "Writing", "KernelPreference", "register_constant", "PassManager is deprecated",
          "AUTOTUNE", "Autotune Choices", "SingleProcess AUTOTUNE", "triton_", "best_kernel",
          "strides:", "dtypes:", "num_choices", "cpuinfo_utils")


def _native_log(f) -> str:
    try:
        f.seek(0)
        lines = f.read().decode("utf-8", "replace").splitlines()
    except Exception:
        return ""
    kept = [ln for ln in lines if ln.strip() and not any(n in ln for n in _NOISE)]
    return "\n".join(kept).strip()


def _executor_main(job_q, res_q):
    """The disposable EXECUTOR child: warm the runtime once, then run jobs off `job_q` until a
    native abort takes it down (→ the coordinator respawns a fresh one) or it gets the None stop
    sentinel. Puts RESULT frames on `res_q` — RAN with raw outputs, or SKIP with the native error."""
    # NOTE: no CUDA_VISIBLE_DEVICES blanking — inherit the (already-pinned) device from the parent.
    from mobile.net.et_runner import run_pte
    try:
        from executorch.runtime import Runtime
        Runtime.get()                                  # warm the kernel registry (incl. CUDA) once
    except Exception:
        pass
    while True:
        item = job_q.get()
        if item is None:
            return
        job_id, pte, inputs = item
        errf = tempfile.TemporaryFile()                # capture native C++ stdout/stderr for SKIPs
        s1, s2 = os.dup(1), os.dup(2)
        os.dup2(errf.fileno(), 1)
        os.dup2(errf.fileno(), 2)
        try:
            frames = P.encode_result(job_id, "RAN", "", run_pte(pte, inputs))
        except Exception as e:
            detail = f"client {type(e).__name__}: {e}"
            native = _native_log(errf)
            if native and native not in detail:
                detail = f"{detail} | {native}"
            frames = P.encode_result(job_id, "SKIP", detail, None)
        finally:
            os.dup2(s1, 1)
            os.dup2(s2, 2)
            os.close(s1)
            os.close(s2)
            errf.close()
        res_q.put(frames)


class Executor:
    """A disposable warm ExecuTorch child + its restart policy — same as openvino_client's. `run()`
    ALWAYS returns RESULT frames: clean run is RAN/SKIP from the child, a native abort → CRASH +
    respawn, a hung kernel → TIMEOUT + respawn."""

    def __init__(self):
        self._mp = mp.get_context("fork")
        self.job_q = self._mp.Queue()
        self.res_q = self._mp.Queue()
        self.proc = None
        self._spawn()

    def _spawn(self):
        self.proc = self._mp.Process(target=_executor_main, args=(self.job_q, self.res_q), daemon=True)
        self.proc.start()
        while True:                                    # drop any stale result from a dead child
            try:
                self.res_q.get_nowait()
            except _queue.Empty:
                break

    def run(self, job_id: str, pte: bytes, inputs: list, timeout: float) -> list:
        self.job_q.put((job_id, pte, inputs))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return self.res_q.get(timeout=0.5)
            except _queue.Empty:
                if not self.proc.is_alive():           # native abort took the child down → CRASH
                    self._spawn()
                    return P.encode_result(job_id, "CRASH", "executor died (native abort)", None)
        if self.proc.is_alive():                       # deadline hit, child still running → hung kernel
            self.proc.terminate()
            self.proc.join()
        self._spawn()
        return P.encode_result(job_id, "TIMEOUT", "kernel exceeded job-timeout", None)

    def close(self):
        self.job_q.put(None)
        if self.proc and self.proc.is_alive():
            self.proc.terminate()


def run_client(host: str, client_port: int, ctrl_port: int, label: str = BACKEND,
               job_timeout: float = 60.0) -> int:
    """Pull jobs from the broker (REQ, LRU), run each in a crash-isolated warm child, return the raw
    outputs. Exits cleanly on the broker's STOP broadcast; zmq auto-reconnects if it restarts."""
    executor = Executor()

    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.HEARTBEAT_IVL, 5000)            # broker reaps us if we die; we detect a dead broker
    req.setsockopt(zmq.HEARTBEAT_TIMEOUT, 20000)
    req.setsockopt(zmq.HEARTBEAT_TTL, 20000)
    req.connect(f"tcp://{host}:{client_port}")
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{host}:{ctrl_port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    poller = zmq.Poller()
    poller.register(req, zmq.POLLIN)
    poller.register(sub, zmq.POLLIN)

    print(f"  {BACKEND}_client[{label}] → broker tcp://{host}:{client_port} "
          f"(warm executor child, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<all>')})",
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
            frames = req.recv_multipart()              # one lean job (pte + inputs)
            try:
                job_id, pte, inputs = P.decode_job(frames)
                result = executor.run(job_id, pte, inputs, job_timeout)
            except Exception as e:
                result = P.encode_result("?", "SKIP", f"client decode {type(e).__name__}: {e}", None)
            req.send_multipart([b"RESULT", *result])
            ran += 1
            if ran % 100 == 0:
                print(f"  {BACKEND}_client[{label}] ran {ran} jobs", flush=True)
    finally:
        executor.close()
        req.close(linger=0)
        sub.close(linger=0)
    print(f"  {BACKEND}_client[{label}] done — ran {ran} jobs", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="CUDA (AOTInductor) ExecuTorch executor client")
    ap.add_argument("--host", default="127.0.0.1", help="broker host")
    ap.add_argument("--client-port", type=int, default=15555)
    ap.add_argument("--ctrl-port", type=int, default=15556)
    ap.add_argument("--label", default=BACKEND)
    ap.add_argument("--gpu", default=None,
                    help="pin this client to one GPU id (sets CUDA_VISIBLE_DEVICES before torch "
                         "loads); omit to use whatever is already visible (delegate targets cuda:0)")
    ap.add_argument("--job-timeout", type=float, default=60.0,
                    help="kill + TIMEOUT a job whose ExecuTorch run exceeds this")
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)  # before any torch import in this process
    return run_client(a.host, a.client_port, a.ctrl_port, a.label, a.job_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
