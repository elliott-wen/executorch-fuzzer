#!/usr/bin/env python3
"""Repro for ethos-u clamp-tensor WRONG-VALUE bug. See _repro_common.py for prerequisites."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _repro_common import run
JOB_ID = "w118:580"
if __name__ == "__main__":
    run(JOB_ID)
