#!/usr/bin/env python3
"""CRASH REPLAY: tensor_util.h scalar_type-mismatch fatal Check.

Reproduces a native SIGABRT (exit 134) from an op that requires its two tensor
operands to share a dtype hitting a fatal ET_CHECK
(`a.scalar_type() == b.scalar_type()`) in `tensor_util.h` instead of a graceful
guard. Eager type-promotes.

Usage: python findings/portable/crash/replay_05-scalar-type-mismatch.py
"""
import os
import subprocess
import sys

REPO = "/data/jwen929/mobile"
JOB = "corpus/portable/w16/w16_1264.job"
EXPECT = 134  # 128+SIGABRT(6); SIGFPE=136, SIGSEGV=139

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


def run_job(job_rel):
    job = os.path.join(REPO, job_rel)
    r = subprocess.run([sys.executable, "-c", CHILD, job],
                       capture_output=True, text=True)
    rc = r.returncode
    lines = r.stderr.strip().splitlines() or [""]
    tail = next((ln for ln in lines if "scalar_type" in ln or "tensor_util" in ln), lines[-1])
    if rc < 0:
        sig = -rc
        code = 128 + sig
        print(f"job {job_rel}  exit={code}  signal={SIGNAMES.get(sig, sig)}")
    else:
        code = rc
        print(f"job {job_rel}  exit={code}  (no fatal signal)")
    print(f"  last stderr: {tail}")
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
