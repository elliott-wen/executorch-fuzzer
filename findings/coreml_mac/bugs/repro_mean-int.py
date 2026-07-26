#!/usr/bin/env python3
"""repro_mean-int.py — Core ML: mean() of integer input is integer-truncated (WRONG-VALUE).

Run: PYTHONPATH=/data/jwen929 .venv/bin/python findings/coreml_mac/bugs/repro_mean-int.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _replay import show

show(["w100:145", "w101:549"],
     note="Core ML mean(int): float mean (eager) -> truncated-toward-zero (device)")
