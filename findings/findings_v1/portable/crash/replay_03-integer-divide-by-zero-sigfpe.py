#!/usr/bin/env python3
"""CRASH REPLAY: integer divide / modulo by zero -> SIGFPE.

Reproduces a native SIGFPE (exit 136 = 128+8) hardware trap from a portable integer
`div`/`remainder`/`fmod`/`floor_divide` kernel performing a raw C++ `/`/`%` on a zero
divisor. There is no abort message (the CPU traps).

Usage: python findings/portable/crash/replay_03-integer-divide-by-zero-sigfpe.py
"""
import os
import subprocess
import sys

REPO = "/data/jwen929/mobile"
JOB = "corpus/portable/w0/w0_626.job"
EXPECT = 136  # 128+SIGFPE(8); SIGABRT=134, SIGSEGV=139

CHILD = r"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
from mobile.executor import corpus, protocol, et_runner
frames = corpus.read_job(sys.argv[1])
job_id, pte, inputs = protocol.decode_job(protocol.job_frames_from_pushjob(frames))
et_runner.run_pte(pte, inputs)     # native trap here = the CRASH
print("RAN OK (no crash)")
"""

SIGNAMES = {6: "SIGABRT", 8: "SIGFPE", 11: "SIGSEGV"}


def run_job(job_rel):
    job = os.path.join(REPO, job_rel)
    r = subprocess.run([sys.executable, "-c", CHILD, job],
                       capture_output=True, text=True)
    rc = r.returncode
    tail = (r.stderr.strip().splitlines() or [""])[-1]
    if rc < 0:
        sig = -rc
        code = 128 + sig
        print(f"job {job_rel}  exit={code}  signal={SIGNAMES.get(sig, sig)}")
    else:
        code = rc
        print(f"job {job_rel}  exit={code}  (no fatal signal)")
    print(f"  last stderr: {tail!r}")
    return rc, code


def main():
    rc, code = run_job(JOB)
    reproduced = rc < 0
    if reproduced:
        match = "matches documented" if code == EXPECT else f"differs from documented {EXPECT}"
        print(f"REPLAY: crash REPRODUCED (exit {code}, {match})")
        return 0
    print("REPLAY: did NOT reproduce (no fatal signal)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
