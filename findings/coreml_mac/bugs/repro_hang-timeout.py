#!/usr/bin/env python3
"""repro_hang-timeout.py — Core ML: ops that wedge (never return) -> TIMEOUT.

Run at window=1; each job holds a worker for the full timeout then records TIMEOUT.
Run: PYTHONPATH=/data/jwen929 .venv/bin/python findings/coreml_mac/bugs/repro_hang-timeout.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _replay import show

show(["w10:113", "w15:609", "w74:426"],
     note="Core ML hangs (expect status=TIMEOUT): slice_scatter / select_scatter / max.unary_out")
