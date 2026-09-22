#!/usr/bin/env python3
"""repro_atan2.py — Core ML: atan2(0, x<0) returns 0 instead of π (WRONG-VALUE).

Run (broker up + Mac CoreML worker connected — see ../README.md):
  PYTHONPATH=/data/jwen929 .venv/bin/python findings/coreml_mac/bugs/repro_atan2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _replay import show

# atan2 graphs where an element has y==0, x<0 (expected π, device gives 0)
show(["w100:421", "w102:358"],
     note="Core ML atan2: π (eager) collapses to 0.0 (device) on the y==0,x<0 branch")
