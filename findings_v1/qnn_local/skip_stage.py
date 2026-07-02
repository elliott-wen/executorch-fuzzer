#!/usr/bin/env python3
"""skip_stage.py — classify why QNN SKIPs a graph, by re-running it through the
qnn_runner.sh seam with run.log preserved and extracting the failing stage + QNN error.

The feeder's TSV only records "rc=2" for a SKIP (qnn_runner.sh's exit code); the actual
cause is in the runner's run.log (our "SKIP: <stage>" line + the QNN [ERROR] line). This
re-runs a sample and emits one tab-separated row per job:

    job_id  <stage>  <qnn_error>

stage  ∈ load_method | alloc_input | alloc_output | set_input | execute | get_outputs | other
Caller must have the QNN env sourced (android-env.sh) + build-x86/lib on LD_LIBRARY_PATH.
Usage: skip_stage.py <jobids_file> [parallel=16]
"""
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

from mobile.net import protocol as P, corpus as C

MOBILE = "/data/jwen929/mobile"
RUNNER_SH = f"{MOBILE}/qnn_client/qnn_runner.sh"

_STAGE = [
    ("load_method", re.compile(r"SKIP: load_method")),
    ("alloc_input", re.compile(r"SKIP: alloc input")),
    ("alloc_output", re.compile(r"SKIP: alloc output")),
    ("set_input", re.compile(r"SKIP: set_input")),
    ("execute", re.compile(r"SKIP: execute")),
    ("get_outputs", re.compile(r"SKIP: get_outputs")),
]
_QNN_ERR = re.compile(r"\[ERROR\] \[Qnn ExecuTorch\]:\s*(?:<E>\s*)?(.+)")
_GEXEC = re.compile(r"qnn_graph_execute failed\. Error (\d+)")


def _classify(run_log: str) -> tuple[str, str]:
    stage = "other"
    for name, rx in _STAGE:
        if rx.search(run_log):
            stage = name
            break
    err = ""
    m = _GEXEC.search(run_log)
    if m:
        err = f"qnn_graph_execute Error {m.group(1)}"
    else:
        # last QNN [ERROR] line (most specific)
        errs = _QNN_ERR.findall(run_log)
        if errs:
            err = errs[-1].strip()[:120]
    return stage, err.replace("\t", " ")


def _one(job_id: str) -> str:
    w, n = job_id.split(":")
    jobf = f"{MOBILE}/corpus_v1/qnn/{w}/{w}_{n}.job"
    try:
        _, pte, inputs = P.decode_job(C.read_job(jobf))
    except Exception as e:
        return f"{job_id}\tdecode_err\t{type(e).__name__}"
    with tempfile.TemporaryDirectory(prefix="skipstage_") as d:
        with open(f"{d}/model.pte", "wb") as f:
            f.write(pte)
        ins = []
        for i, t in enumerate(inputs):
            _, raw = P.tensor_to_meta_blob(t.contiguous())
            ip = f"{d}/in_{i}.bin"
            with open(ip, "wb") as f:
                f.write(raw)
            ins += ["--input", ip]
        cmd = ["bash", RUNNER_SH, "--pte", f"{d}/model.pte", "--out", d, "--soc", "SM8450", *ins]
        try:
            subprocess.run(cmd, capture_output=True, timeout=240)
        except subprocess.TimeoutExpired:
            return f"{job_id}\ttimeout\t"
        run_log = ""
        try:
            with open(f"{d}/run.log") as f:
                run_log = f.read()
        except OSError:
            return f"{job_id}\tno_runlog\t"
        stage, err = _classify(run_log)
        return f"{job_id}\t{stage}\t{err}"


def main() -> int:
    jobids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    par = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    with ProcessPoolExecutor(max_workers=par) as ex:
        for row in ex.map(_one, jobids):
            print(row, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
