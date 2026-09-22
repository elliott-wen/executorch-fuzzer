#!/usr/bin/env python3
"""repro.py — self-contained device repro for any confirmed OpenVINO single-op finding.

Given one or more corpus job_ids, this reruns the exact stored one-op .pte on the live
OpenVINO host runtime (through the broker) and prints eager (CPU reference) vs device output,
so the mismatch / crash / skip is reproduced and pinned. Every job below is a single operator
wired to leaf inputs, so the printed divergence *is* the localized finding.

Prereqs (a human sets these up — see findings/openvino_single/README.md):
  1. broker:  PYTHONPATH=/data/jwen929 .venv/bin/python -m mobile broker \
                 --job-port 15664 --client-port 15665 --ctrl-port 15666
  2. worker:  MOBILE_BACKENDS=openvino PYTHONPATH=/data/jwen929 CUDA_VISIBLE_DEVICES= \
                 .venv/bin/python openvino_client/openvino_client.py \
                 --client-port 15665 --ctrl-port 15666 --job-timeout 120

Run:
  PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/bugs/repro.py [job_id ...]
With no args it runs one representative job per confirmed mechanism.
"""
import sys, json, subprocess
from pathlib import Path

ROOT = Path("/data/jwen929/mobile")
CORPUS = str(ROOT / "corpus_v3/openvino")

# representative confirmed jobs, keyed by mechanism (device-verified, REPRO N=5)
DEFAULT = {
    "ZEROED clone (float32)":            "w0:144",
    "ZEROED clone (bool)":               "w0:375",
    "ZEROED alias_copy (int64)":         "w0:260",
    "ZEROED lift_fresh_copy":            "w0:32",
    "ZEROED t_copy (float16)":           "w0:183",
    "ZEROED transpose_copy.int":         "w0:270",
    "WRONG-VALUE diagonal_copy (row!=diag)": "w0:329",
    "WRONG-VALUE unfold_copy (raw order)":   "w0:220",
    "WRONG-VALUE reflection_pad2d":      "w0:121",
    "NONFINITE batch_norm.no_stats":     "w106:221",
    "WRONG-SHAPE min.dim (out= empty)":  "w0:115",
    "WRONG-SHAPE max.dim (out= empty)":  "w10:66",
    "VALUE bitwise_left_shift (UB shift)": "w103:452",
}


def run(jobs):
    cmd = [f"{ROOT}/.venv/bin/python", "-m", "mobile", "feed", "--corpus", CORPUS,
           "--job-port", "15664", "--ctrl-port", "15666", "--timeout", "60"] + jobs
    p = subprocess.run(cmd, capture_output=True, text=True, cwd="/data/jwen929",
                       env={"PYTHONPATH": "/data/jwen929", "PATH": f"{ROOT}/.venv/bin:/usr/bin:/bin"})
    out = {}
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                o = json.loads(line); out[o["job_id"]] = o
            except Exception:
                pass
    return out


def show(label, job, o):
    print(f"\n### {label}   [{job}]")
    if not o:
        print("  (no result — is the broker+worker up?)"); return
    print(f"  op      : {o.get('op_chain')}")
    print(f"  status  : {o.get('status')}   {o.get('detail','')}")
    outs = o.get("outputs") or []
    if outs:
        out = outs[0]
        print(f"  eager   : {str(out.get('eager'))[:72]}")
        print(f"  device  : {str(out.get('et'))[:72]}")
        if "et_shape" in out:
            print(f"  shapes  : eager {out.get('shape')}  vs  device {out.get('et_shape')}")


def main():
    args = sys.argv[1:]
    if args:
        res = run(args)
        for j in args:
            show(j, j, res.get(j))
    else:
        jobs = list(DEFAULT.values())
        res = run(jobs)
        for label, j in DEFAULT.items():
            show(label, j, res.get(j))


if __name__ == "__main__":
    main()
