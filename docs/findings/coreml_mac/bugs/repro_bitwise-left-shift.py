#!/usr/bin/env python3
"""repro_bitwise-left-shift.py — Core ML: integer left-shift overflow -> wrong int / -inf.

Run: PYTHONPATH=/data/jwen929 .venv/bin/python findings/coreml_mac/bugs/repro_bitwise-left-shift.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _replay import show

show(["w1:321", "w100:28"],
     note="Core ML bitwise_left_shift: int64->-2^31 overflow (w1:321); int32->-inf (w100:28)")
