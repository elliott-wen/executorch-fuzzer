#!/usr/bin/env python3
"""CRASH REPLAY: narrow_copy / select with negative or out-of-range `dim`.

Reproduces a native SIGABRT (exit 134) from the portable `narrow_copy` validator
calling `in.size(dim)` with a raw negative dim, firing a fatal ET_CHECK in
`tensor_impl.h:136` ("... got -1").

Usage: python findings/portable/crash/replay_01-narrow_copy-negative-dim.py
"""
import os
import subprocess
import sys

REPO = "/data/jwen929/mobile"
JOB = "corpus/portable/w0/w0_1693.job"
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
    tail = (r.stderr.strip().splitlines() or [""])[-1]
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
