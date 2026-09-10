#!/usr/bin/env python3
"""CRASH REPLAY: memory corruption (the most severe class).

Iterates over all listed jobs and asserts each crashes with a fatal signal.
These are out-of-bounds writes / heap corruption (double free, free(): invalid
next size, SIGSEGV) -- a kernel scribbled past a buffer before any check fired.

Usage: python findings/portable/crash/replay_08-memory-corruption.py
"""
import os
import subprocess
import sys

REPO = "/data/jwen929/mobile"

# (job, documented expected exit code)
JOBS = [
    ("corpus/portable/w56/w56_557.job", 134),   # double free or corruption (out)
    ("corpus/portable/w92/w92_1487.job", 134),  # free(): invalid next size (fast)
    ("corpus/portable/w10/w10_359.job", 139),   # SIGSEGV (no message)
    ("corpus/portable/w38/w38_1077.job", 139),  # SIGSEGV (no message)
]

CHILD = r"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
from mobile.net import corpus, protocol, et_runner
frames = corpus.read_job(sys.argv[1])
job_id, pte, inputs = protocol.decode_job(protocol.job_frames_from_pushjob(frames))
et_runner.run_pte(pte, inputs)     # native abort here = the CRASH
print("RAN OK (no crash)")
"""

SIGNAMES = {6: "SIGABRT", 8: "SIGFPE", 11: "SIGSEGV"}


def run_job(job_rel, expect):
    job = os.path.join(REPO, job_rel)
    r = subprocess.run([sys.executable, "-c", CHILD, job],
                       capture_output=True, text=True)
    rc = r.returncode
    tail = (r.stderr.strip().splitlines() or [""])[-1]
    if rc < 0:
        sig = -rc
        code = 128 + sig
        print(f"job {job_rel}  exit={code}  signal={SIGNAMES.get(sig, sig)}  (documented {expect})")
    else:
        code = rc
        print(f"job {job_rel}  exit={code}  (no fatal signal; documented {expect})")
    print(f"  last stderr: {tail}")
    return rc < 0


def main():
    results = [run_job(job, expect) for job, expect in JOBS]
    all_repro = all(results)
    n_ok = sum(results)
    print(f"REPLAY: {n_ok}/{len(JOBS)} jobs crashed (fatal signal)")
    if all_repro:
        print("REPLAY: ALL memory-corruption jobs REPRODUCED")
        return 0
    print("REPLAY: NOT all jobs reproduced")
    return 1


if __name__ == "__main__":
    sys.exit(main())
