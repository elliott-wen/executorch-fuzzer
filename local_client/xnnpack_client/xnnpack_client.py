"""xnnpack_client.py — XNNPACK EXECUTOR worker (the host analog of the Android client).

Same role as the on-device Android client, the FVP/QNN clients, and the old net/client.py:
pull `.pte` jobs from the broker, run them, and return the RAW output tensors over the
language-neutral binary protocol. The feeder still owns the eager reference and does the
diff — this process is a thin executor, exactly like the phone.

What's special here: the `.pte` was lowered to the **XNNPACK** delegate (Google's
optimized CPU kernels for fp32/quantized ops), and it runs *in-process* on the ExecuTorch
**host runtime** (`executorch.runtime`), which loads the XNNPACK backend the delegate's
blobs call into. There is no external simulator to shell out to (unlike fvp_client/
qnn_client); the runtime IS the device seam. Portable-backend `.pte`s also run on this
same in-process runtime, so they need no separate client.

Because a hard kernel failure in the ExecuTorch runtime is a NATIVE abort that takes the
whole process down, the run is crash-isolated the SAME way the phone is: a stable
COORDINATOR (this process, holds the broker connection) drives a disposable warm EXECUTOR
child (a forked process that holds the runtime). A native abort kills only the child →
the coordinator reports CRASH and respawns it; a hung kernel is killed at the deadline →
TIMEOUT + respawn. So the worker ALWAYS returns a verdict (RAN/SKIP/CRASH/TIMEOUT) —
nothing relies on the broker detecting a disconnect.

    JOB(pte+inputs) ─► warm fork runs forward(*inputs) on the ET runtime (XNNPACK delegate)
                    ─► RESULT(raw outputs) ─► broker ─► feeder diffs vs eager

Usage:
    python xnnpack_client/xnnpack_client.py --host <broker-ip> [--job-timeout 30]
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU-only; set before torch is imported

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

from mobile.executor import protocol as P                  # noqa: E402

BACKEND = "xnnpack"

# Known-NOISE substrings: ARM cpuinfo probing on x86 + a torch deprecation warning. Unconditional
# startup chatter, never part of an actual failure — dropped, but EVERYTHING else is preserved.
_NOISE = ("cpuinfo_utils", "KernelPreference", "register_constant",
          "Reading file", "Failed to open midr", "CPU info and manual query")


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
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from mobile.executor.et_runner import run_pte
    try:
        from executorch.runtime import Runtime
        Runtime.get()                                  # warm the kernel registry (incl. XNNPACK) once
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
    """A disposable warm ExecuTorch child + its restart policy — the host analog of the phone's
    Lane↔executor split. `run()` ALWAYS returns RESULT frames: a clean run is RAN/SKIP from the
    child, a native abort → CRASH + respawn, a hung kernel → TIMEOUT + respawn. So the worker never
    needs the broker to infer a crash from a dropped connection."""

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
               job_timeout: float = 30.0) -> int:
    """Pull jobs from the broker (REQ, LRU), run each in a crash-isolated warm child, return the
    raw outputs. Exits cleanly on the broker's STOP broadcast; zmq auto-reconnects if it restarts."""
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

    print(f"  {BACKEND}_client[{label}] → broker tcp://{host}:{client_port} (warm executor child)",
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
    ap = argparse.ArgumentParser(description="XNNPACK ExecuTorch executor client")
    ap.add_argument("--host", default="127.0.0.1", help="broker host")
    ap.add_argument("--client-port", type=int, default=15555)
    ap.add_argument("--ctrl-port", type=int, default=15556)
    ap.add_argument("--label", default=BACKEND)
    ap.add_argument("--job-timeout", type=float, default=30.0,
                    help="kill + TIMEOUT a job whose ExecuTorch run exceeds this")
    a = ap.parse_args()
    return run_client(a.host, a.client_port, a.ctrl_port, a.label, a.job_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
