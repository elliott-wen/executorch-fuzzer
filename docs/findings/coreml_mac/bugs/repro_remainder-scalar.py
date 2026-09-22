#!/usr/bin/env python3
"""repro_remainder-scalar.py — Core ML: remainder.Scalar returns all-zeros (ZEROED).

Run: PYTHONPATH=/data/jwen929 .venv/bin/python findings/coreml_mac/bugs/repro_remainder-scalar.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _replay import show

show(["w14:186", "w35:750"],
     note="Core ML remainder.Scalar: nonzero remainder (eager) -> all-zeros (device)")
