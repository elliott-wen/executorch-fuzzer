"""coreml_client.py — CoreML (Apple) EXECUTOR worker (a host-runtime client like xnnpack_client).

Same role as the on-device Android client, the FVP/QNN clients, and the portable/xnnpack/
openvino clients: pull `.pte` jobs from the broker, run them, and return the RAW output
tensors over the language-neutral binary protocol. The feeder still owns the eager reference
and does the diff — this process is a thin executor, exactly like the phone.

What's special here: the `.pte` was lowered to the **CoreML** delegate (Apple; see
generator/lower/backends/coreml.py), and it runs *in-process* on the ExecuTorch **host runtime**
(`executorch.runtime`), which loads the CoreML backend the delegate's blobs call into. On this
Mac the delegate dispatches to the Apple **Neural Engine / GPU / CPU** via CoreML.framework —
there is no external simulator to shell out to (unlike fvp_client/qnn_client); the runtime IS
the device seam. Mechanically the run path is IDENTICAL to the xnnpack/openvino clients — the
difference is purely which backend the program was compiled for — so this stays a separate
folder only to keep one self-contained client per backend.

This requires the ExecuTorch runtime built with the CoreML backend registered (the macOS pip
wheel ships it: `Runtime.get().backend_registry` lists `CoreMLBackend`). CoreML *compiles* each
new program to an `.mlmodelc` on first `load_method` (native log `ETCoreMLModelManager.mm ...
Compiling with a 5 min timeout`), so a fresh program's first run is slow — hence the generous
default --job-timeout. Nothing about coremltools is needed at RUN time; that is an export-time
dep and the `.pte` here is already lowered.

macOS PROCESS MODEL: the sibling clients FORK their warm executor child, but on macOS forking
this coordinator (multithreaded via zmq/torch, and it has touched ObjC through Apple-linked libs)
aborts — "+[NSNumber initialize] may have been in progress in another thread when fork() was
called. Crashing instead." — and OBJC_DISABLE_INITIALIZE_FORK_SAFETY isn't reliably honored when
set from within Python (libobjc reads it before any Python runs). So this client uses SPAWN: each
executor child is a clean fresh interpreter — no inherited threads, no half-initialized ObjC — and
CoreML is only ever touched there, never in the coordinator. The tradeoff is that a spawned child
re-imports torch/executorch, but that cost is paid only per recycle/respawn, not per job.

Because a hard kernel failure in the ExecuTorch runtime is a NATIVE abort that takes the whole
process down, the run is crash-isolated the SAME way the phone is: a stable COORDINATOR (this
process, holds the broker connection) drives a disposable warm EXECUTOR child (a spawned process
that holds the runtime). A native abort kills only the child → the coordinator reports CRASH and
respawns it; a hung kernel is killed at the deadline → TIMEOUT + respawn. So the worker ALWAYS
returns a verdict (RAN/SKIP/CRASH/TIMEOUT) — nothing relies on the broker detecting a disconnect.

LIFETIME BOUND (CoreML-specific leak): the ExecuTorch CoreML backend RETAINS every compiled
model in an in-process asset manager (native log `... Transferring ownership to assetManager`).
A fuzzer feeds a fresh graph each job, so each job compiles a NEW `.mlmodelc` the warm child
never releases — fds / memory / XPC handles to the CoreML daemon climb until `load_method`
starts native-aborting and the child then CRASHes EVERY job. So we cap the child's lifetime:
the coordinator recycles it every `--recycle-every` jobs (default 100), and immediately if
`--max-consecutive-crashes` non-RAN results in a row show the runtime already wedged. A recycle
is a cheap child respawn + one `Runtime.get()` warm-up — it drops the whole asset manager. For a rarer
OS/daemon-level wedge that survives a fresh child, `--restart-every N` re-execs the entire
process (new XPC session, new fd table, fresh runtime).

JOB(pte+inputs) ─► warm child runs forward(*inputs) on the ET runtime (CoreML delegate → ANE/GPU/CPU)
─► RESULT(raw outputs) ─► broker ─► feeder diffs vs eager
(… every --recycle-every jobs the warm child is torn down + respawned to drop CoreML's cache)

NUMERICS NOTE (same caveat as fvp_client/qnn_client): CoreML computes in fp16 on the ANE/GPU, so
outputs come back with quantization/precision error vs an fp32 eager oracle (a trivial matmul+relu
already differs by ~3e-3). Choosing the right reference/tolerance is a feeder/executor.compare concern,
not this client's — this client just returns what CoreML produced.

Usage:
python local_client/coreml_client/coreml_client.py --host <broker-ip> [--job-timeout 120] [--recycle-every 100]
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU/ANE/GPU only; set before torch imports
# Belt-and-suspenders for any stray fork (Apple frameworks abort if an ObjC class initialized in
# a parent is used across a fork). The executor child uses SPAWN (see Executor.__init__), which
# avoids this entirely; this env var is a fallback and is harmless under spawn.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import argparse
import json
import multiprocessing as mp
import queue as _queue
import sys
import tempfile
import time
from pathlib import Path

import zmq

HERE = Path(__file__).resolve().parent
IMPORT_ROOT = HERE.parent.parent.parent              # dir on sys.path so `import mobile` resolves
sys.path.insert(0, str(IMPORT_ROOT))

from mobile.executor import protocol as P            # noqa: E402

BACKEND = "coreml"

# Known-NOISE substrings: unconditional startup chatter that is never part of an actual failure —
# dropped, but EVERYTHING else is preserved. Covers CoreML's native model-compilation logging
# (each fresh program compiles to an .mlmodelc on first load) plus torch/cpuinfo warm-up lines.
_NOISE = ("cpuinfo_utils", "KernelPreference", "register_constant",
          "Reading file", "Failed to open midr", "CPU info and manual query",
          "ETCoreMLModelManager.mm", "Cache Miss", "not pre-compiled",
          "Compiling with a", "Successfully got compiled model", "Transferring ownership",
          "coreml_preprocess", "Redirects are currently not supported",
          "MIL frontend_pytorch pipeline", "MIL default pipeline", "MIL backend_mlprogram")


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
    sentinel. Puts RESULT frames on `res_q` — RAN with raw outputs, or SKIP with the native error.

    CoreML is first touched HERE (in the spawned child), never in the coordinator."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    from mobile.executor.et_runner import run_pte
    try:
        from executorch.runtime import Runtime
        Runtime.get()                              # warm the kernel registry (incl. CoreML) once
    except Exception:
        pass
    while True:
        item = job_q.get()
        if item is None:
            return
        job_id, pte, inputs = item
        errf = tempfile.TemporaryFile()             # capture native C++ stdout/stderr for SKIPs
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
    child, a native abort → CRASH + respawn, a hung kernel → TIMEOUT + respawn. So the worker
    never needs the broker to infer a crash from a dropped connection."""

    def __init__(self):
        # SPAWN, not fork: forking this coordinator (multithreaded via zmq/torch, and it has
        # touched ObjC through the Apple-linked libs) aborts on macOS — "+[NSNumber initialize]
        # may have been in progress in another thread when fork() was called. Crashing instead."
        # OBJC_DISABLE_INITIALIZE_FORK_SAFETY doesn't reliably help (libobjc reads it before any
        # Python runs). Spawn boots a clean interpreter, dodging it entirely; the extra import
        # cost is paid only per recycle/respawn (~every recycle_every jobs), not per job.
        self._mp = mp.get_context("spawn")
        self.proc = None
        self.job_q = None
        self.res_q = None
        self._spawn()

    def _spawn(self):
        # FRESH queues every child. Never reuse a Queue across a terminate(): if the child was
        # blocked in job_q.get() (its normal idle state) when SIGTERM hit, it died holding the
        # queue's internal reader lock, which is then never released — the NEXT child's get()
        # would block forever (the "stall after recycle"). New queues sidestep the poisoned lock;
        # the old ones are closed in _kill() and garbage-collected.
        self.job_q = self._mp.Queue()
        self.res_q = self._mp.Queue()
        self.proc = self._mp.Process(target=_executor_main, args=(self.job_q, self.res_q),
                                     daemon=True)
        self.proc.start()

    def _kill(self):
        """Stop the current child and release its (possibly lock-poisoned) queues."""
        if self.proc and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()
        for q in (self.job_q, self.res_q):
            if q is not None:
                q.close()                  # drop the coordinator-side feeder thread + fds
        self.proc = self.job_q = self.res_q = None

    def run(self, job_id: str, pte: bytes, inputs: list, timeout: float) -> list:
        self.job_q.put((job_id, pte, inputs))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return self.res_q.get(timeout=0.5)
            except _queue.Empty:
                if not self.proc.is_alive():   # native abort took the child down → CRASH
                    self.recycle()
                    return P.encode_result(job_id, "CRASH", "executor died (native abort)", None)
        self.recycle()                         # deadline hit → kill the hung child, fresh one
        return P.encode_result(job_id, "TIMEOUT", "kernel exceeded job-timeout", None)

    def recycle(self):
        """Tear down the warm child and start a fresh one (with fresh queues). This is the lever
        against CoreML's unbounded compiled-model retention: killing the child drops its asset
        manager (every `.mlmodelc` it ever loaded) and all the fds / XPC handles that came with
        them. Cheap — a spawn + one `Runtime.get()` warm-up; the broker connection is untouched."""
        self._kill()
        self._spawn()

    def close(self):
        self._kill()


def _result_status(frames: list) -> str:
    """Peek the verdict ('RAN'/'SKIP'/'CRASH'/'TIMEOUT') from a result's JSON header (frame 0),
    so the coordinator can spot a crash cascade without decompressing any payload."""
    try:
        return json.loads(frames[0].decode("utf-8")).get("status", "?")
    except Exception:
        return "?"


def run_client(host: str, client_port: int, ctrl_port: int, label: str = BACKEND,
               job_timeout: float = 120.0, recycle_every: int = 100,
               max_consecutive_crashes: int = 20, restart_every: int = 0) -> int:
    """Pull jobs from the broker (REQ, LRU), run each in a crash-isolated warm child, return the
    raw outputs. Exits cleanly on the broker's STOP broadcast; zmq auto-reconnects if it restarts.

    The warm child is recycled every `recycle_every` jobs (and immediately after
    `max_consecutive_crashes` non-RAN results in a row) so CoreML's ever-growing compiled-model
    cache can't wedge it into crashing every job; `restart_every` re-execs the whole process."""
    executor = Executor()

    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.HEARTBEAT_IVL, 5000)    # broker reaps us if we die; we detect a dead broker
    req.setsockopt(zmq.HEARTBEAT_TIMEOUT, 20000)
    req.setsockopt(zmq.HEARTBEAT_TTL, 20000)
    req.connect(f"tcp://{host}:{client_port}")
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{host}:{ctrl_port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    poller = zmq.Poller()
    poller.register(req, zmq.POLLIN)
    poller.register(sub, zmq.POLLIN)

    print(f"  {BACKEND}_client[{label}] → broker tcp://{host}:{client_port} (warm executor child, "
          f"ANE/GPU/CPU; recycle every {recycle_every or '∞'})", flush=True)
    req.send_multipart([b"READY"])
    ran = 0
    consec_bad = 0                             # non-RAN results in a row (crash-cascade guard)
    try:
        while True:
            socks = dict(poller.poll(timeout=1000))
            if sub in socks and sub.recv() == b"STOP":
                break
            if req not in socks:
                continue
            frames = req.recv_multipart()      # one lean job (pte + inputs)
            try:
                job_id, pte, inputs = P.decode_job(frames)
                result = executor.run(job_id, pte, inputs, job_timeout)
            except Exception as e:
                result = P.encode_result("?", "SKIP", f"client decode {type(e).__name__}: {e}",
                                         None)
            req.send_multipart([b"RESULT", *result])
            ran += 1
            consec_bad = 0 if _result_status(result) == "RAN" else consec_bad + 1

            # Bound the warm child's life so CoreML's retained-model growth can't wedge it into
            # crashing every job. Proactive recycle on a schedule; early recycle if a cascade of
            # non-RAN results says it's already wedged before the next scheduled one.
            if recycle_every and ran % recycle_every == 0:
                executor.recycle()
                consec_bad = 0
                print(f"  {BACKEND}_client[{label}] recycled child (scheduled, {ran} jobs)",
                      flush=True)
            elif max_consecutive_crashes and consec_bad >= max_consecutive_crashes:
                executor.recycle()
                print(f"  {BACKEND}_client[{label}] recycled child (circuit-breaker: "
                      f"{consec_bad} non-RAN in a row)", flush=True)
                consec_bad = 0

            if ran % 100 == 0:
                print(f"  {BACKEND}_client[{label}] ran {ran} jobs", flush=True)

            # Nuclear reset for an OS/daemon-level wedge a fresh child can't clear: re-exec the
            # whole process (new XPC session, fd table, runtime). Off by default (restart_every=0).
            if restart_every and ran % restart_every == 0:
                print(f"  {BACKEND}_client[{label}] re-exec process (fresh runtime, {ran} jobs)",
                      flush=True)
                executor.close()
                req.close(linger=0)
                sub.close(linger=0)
                ctx.term()
                os.execv(sys.executable,
                         [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    finally:
        executor.close()
        req.close(linger=0)
        sub.close(linger=0)
    print(f"  {BACKEND}_client[{label}] done — ran {ran} jobs", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="CoreML (Apple) ExecuTorch executor client")
    ap.add_argument("--host", default="127.0.0.1", help="broker host")
    ap.add_argument("--client-port", type=int, default=15555)
    ap.add_argument("--ctrl-port", type=int, default=15556)
    ap.add_argument("--label", default=BACKEND)
    ap.add_argument("--job-timeout", type=float, default=120.0,
                    help="kill + TIMEOUT a job whose ExecuTorch run exceeds this; CoreML compiles "
                         "a fresh program on first load (up to a 5-min native timeout), so keep "
                         "it generous")
    ap.add_argument("--recycle-every", type=int, default=100,
                    help="respawn the warm executor child every N jobs to drop CoreML's retained "
                         "compiled-model cache before it wedges the runtime (0 disables)")
    ap.add_argument("--max-consecutive-crashes", type=int, default=20,
                    help="recycle the child early if this many jobs in a row return non-RAN "
                         "(CRASH/TIMEOUT/SKIP) — catches a wedged runtime before the next "
                         "scheduled recycle (0 disables)")
    ap.add_argument("--restart-every", type=int, default=0,
                    help="re-exec the whole process every N jobs for a total reset (fresh XPC "
                         "session, fds, runtime); 0 disables. Use only if child recycling proves "
                         "insufficient")
    a = ap.parse_args()
    return run_client(a.host, a.client_port, a.ctrl_port, a.label, a.job_timeout,
                      a.recycle_every, a.max_consecutive_crashes, a.restart_every)


if __name__ == "__main__":
    raise SystemExit(main())
