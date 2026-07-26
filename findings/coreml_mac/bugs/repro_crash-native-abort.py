#!/usr/bin/env python3
"""repro_crash-native-abort.py — Core ML: delegated ops that hard-abort the runtime.

Each job below reproduced `executor died (native abort)` on a clean window=1 re-run (and N=5
for the gated ones). Run at window=1 so one abort does not take out the others.

Run: PYTHONPATH=/data/jwen929 .venv/bin/python findings/coreml_mac/bugs/repro_crash-native-abort.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _replay import show

# one confirmed crashing job per operator
JOBS = {
    "unfold_copy":            "w1:781",
    "prod.int_out":           "w100:32",
    "diagonal_copy":          "w109:182",
    "scatter.value_out":      "w101:236",
    "linear.out":             "w111:257",
    "mean.out":               "w82:329",
    "cumsum.out":             "w48:376",
    "logical_or.out":         "w109:217",
    "mm.out":                 "w79:455",
    "t_copy":                 "w13:298",
    "tril.out":               "w35:277",
    "alias_copy":             "w113:699",
    "view_copy":              "w22:560",
}
jobs = [j for j in JOBS.values() if j]
show(jobs, note="Core ML native-abort crashes (expect status=CRASH 'executor died (native abort)')")
