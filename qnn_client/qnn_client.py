"""qnn_client.py — QNN / Hexagon-HTP EXECUTOR worker (the emulator analog of net/client.py).

Same role as the on-device Android client, the Ethos-U `fvp_client`, and the local
`mobile client`: pull `.pte` jobs from the broker, run them, return the RAW output tensors
over the language-neutral binary protocol. The feeder still holds the eager reference and
does the diff — this process is a thin executor, exactly like the phone.

The ONLY thing that differs from net/client.py is *where* the `.pte` runs. There is no
ExecuTorch host runtime for a QNN HTP context binary, so instead of `executorch.runtime`
execution, each job is run on the **Qualcomm HTP x86 emulator** via a small shell seam
(`qnn_runner.sh`), which shells out to the SDK-linked `qnn_executor_runner`:

    JOB(pte+inputs) ─► write pte + in_<i>.bin ─► qnn_runner.sh ─► HTP emulator runs the .pte
                    ◄─ output_0_<i>.raw ◄──────────────────────────────────────────────────
    ─► RESULT(raw outputs) ─► broker ─► feeder diffs vs eager

`qnn_executor_runner` writes bare raw output bytes with NO dtype/shape, so — unlike the FVP
seam — we recover the output {dtype, shape} from the `.pte` itself via the ExecuTorch
runtime's `method_meta("forward")`. That is a pure flatbuffer parse (it does NOT load_method,
so it never initializes the QNN delegate or touches the emulator) and gives exactly the
program output order qnn_executor_runner numbers its files by.

Crash/timeout isolation mirrors net/client.py: the runner runs as a child process; a
non-zero exit → CRASH, exceeding the deadline → TIMEOUT. The worker ALWAYS returns a verdict.

NUMERICS NOTE (same caveat as fvp_client): the QNN float path is fp16 HTP (see
gen/backends/qualcomm.py USE_FP16), so outputs come back fp16 and carry quantization/precision
error vs an fp32 eager oracle. Choosing the right reference is a feeder/compare.py concern,
not this client's — this client just returns what the emulator produced.

Usage:
    python qnn_client/qnn_client.py --host <broker-ip> --soc SM8450
    python qnn_client/qnn_client.py --selftest          # plumbing test, no QNN SDK needed
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU-only; never touch CUDA

import argparse
import glob
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import zmq

HERE = Path(__file__).resolve().parent
IMPORT_ROOT = HERE.parent.parent                      # dir on sys.path so `import mobile` resolves
sys.path.insert(0, str(IMPORT_ROOT))

from mobile.net import protocol as P                  # noqa: E402

QNN_RUNNER = HERE / "qnn_runner.sh"


def _output_metas(pte: bytes) -> list[dict]:
    """Recover [{dtype, dims}, ...] for the program's outputs from the .pte, in order.

    Uses ExecuTorch's method_meta — a parse of the program's I/O specs that does NOT
    load_method, so the QNN delegate is never initialized and no emulator is needed here.
    `dtype` is the ScalarType int code, exactly the protocol's tensor dtype encoding."""
    from executorch.runtime import Runtime

    prog = Runtime.get().load_program(pte)
    mm = prog.method_meta("forward")
    metas = []
    for i in range(mm.num_outputs()):
        ts = mm.output_tensor_meta(i)
        dt = ts.dtype()
        # ExecuTorch ScalarType ints match torch/c10 ScalarType ints, which is what the
        # protocol's dtype<->code maps use. int(dt) is robust across pybind enum shapes.
        code = int(dt) if not isinstance(dt, int) else dt
        metas.append({"dtype": code, "dims": list(ts.sizes())})
    return metas


class QnnExecutor:
    """Runs one `.pte` on the QNN HTP emulator and returns its outputs. Always returns RESULT
    frames: RAN with outputs, or SKIP/CRASH/TIMEOUT with a detail string (no outputs)."""

    def __init__(self, soc: str, runner_bin: str | None, sdk: str | None,
                 et_build: str | None, selftest: bool = False):
        self.soc = soc
        self.runner_bin = runner_bin
        self.sdk = sdk
        self.et_build = et_build
        self.selftest = selftest

    def run(self, job_id: str, pte: bytes, inputs: list, timeout: float) -> list:
        if self.selftest:
            # Plumbing test: echo inputs back as outputs so the broker/feeder path can be
            # exercised end-to-end without a QNN SDK. (Will MISMATCH; that's fine.)
            return P.encode_result(job_id, "RAN", "selftest-echo", [t.clone() for t in inputs])

        # Output {dtype, shape} come from the program, not the runner's bare .raw files.
        try:
            out_metas = _output_metas(pte)
        except Exception as e:
            return P.encode_result(job_id, "SKIP", f"method_meta {type(e).__name__}: {e}", None)

        with tempfile.TemporaryDirectory(prefix="qnn_job_") as d:
            work = Path(d)
            pte_path = work / "model.pte"
            pte_path.write_bytes(pte)
            cmd = [str(QNN_RUNNER), "--pte", str(pte_path), "--out", str(work),
                   "--soc", self.soc]
            for i, t in enumerate(inputs):
                _, raw = P.tensor_to_meta_blob(t.contiguous())
                ip = work / f"in_{i}.bin"
                ip.write_bytes(raw)
                cmd += ["--input", str(ip)]
            if self.runner_bin:
                cmd += ["--runner", self.runner_bin]
            if self.sdk:
                cmd += ["--sdk", self.sdk]
            if self.et_build:
                cmd += ["--et-build", self.et_build]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return P.encode_result(job_id, "TIMEOUT", "qnn emulator exceeded job-timeout", None)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-300:]
                # exit 2 = graph not runnable on this target → SKIP; anything else → CRASH.
                status = "SKIP" if proc.returncode == 2 else "CRASH"
                return P.encode_result(job_id, status, f"qnn_runner rc={proc.returncode}: {tail}", None)

            # qnn_executor_runner names files output_<inference>_<output>.raw; one input
            # line ⇒ inference 0. Pair them with the program's output metas, in order.
            raws = sorted(glob.glob(str(work / "output_0_*.raw")),
                          key=lambda p: int(Path(p).stem.rsplit("_", 1)[1]))
            if len(raws) != len(out_metas):
                return P.encode_result(
                    job_id, "SKIP",
                    f"output count {len(raws)} != method_meta {len(out_metas)}", None)
            try:
                outs = [P.tensor_from_meta_blob(m, Path(r).read_bytes())
                        for m, r in zip(out_metas, raws)]
            except Exception as e:
                return P.encode_result(job_id, "SKIP", f"output decode {type(e).__name__}: {e}", None)
            return P.encode_result(job_id, "RAN", "", outs)

    def close(self):
        pass


def run_client(host: str, client_port: int, ctrl_port: int, executor: QnnExecutor,
               label: str, job_timeout: float) -> int:
    """LRU work-pull from the broker, identical handshake to net/client.py and fvp_client.py."""
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

    print(f"  qnn_client[{label}] → broker tcp://{host}:{client_port} (soc={executor.soc})",
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
                result = executor.run(job_id, pte, inputs, job_timeout)
            except Exception as e:
                result = P.encode_result("?", "SKIP", f"client decode {type(e).__name__}: {e}", None)
            req.send_multipart([b"RESULT", *result])
            ran += 1
            if ran % 50 == 0:
                print(f"  qnn_client[{label}] ran {ran} jobs", flush=True)
    finally:
        executor.close()
        req.close(linger=0)
        sub.close(linger=0)
    print(f"  qnn_client[{label}] done — ran {ran} jobs", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Qualcomm HTP emulator executor client")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--client-port", type=int, default=15555)
    ap.add_argument("--ctrl-port", type=int, default=15556)
    ap.add_argument("--label", default="qnn")
    ap.add_argument("--job-timeout", type=float, default=120.0,
                    help="per-job wall-clock budget (emulator runs are slow); TIMEOUT past this")
    ap.add_argument("--soc", default="SM8450", help="Snapdragon SoC the .pte was compiled for")
    ap.add_argument("--runner", default=None,
                    help="x86 qnn_executor_runner binary (else qnn_runner.sh autodetects)")
    ap.add_argument("--sdk", default=None,
                    help="QNN_SDK_ROOT (else from env / qnn_runner.sh)")
    ap.add_argument("--et-build", default=None,
                    help="ExecuTorch x86 build dir holding libqnn_executorch_backend.so + runner")
    ap.add_argument("--selftest", action="store_true",
                    help="echo inputs as outputs (exercise the broker path without the QNN SDK)")
    a = ap.parse_args()
    ex = QnnExecutor(a.soc, a.runner, a.sdk, a.et_build, selftest=a.selftest)
    return run_client(a.host, a.client_port, a.ctrl_port, ex, a.label, a.job_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
