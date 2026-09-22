#!/usr/bin/env python3
"""repro_upsample-bilinear2d-aa.py — Core ML: anti-aliased bilinear resize is wrong (WRONG-VALUE).

Run: PYTHONPATH=/data/jwen929 .venv/bin/python findings/coreml_mac/bugs/repro_upsample-bilinear2d-aa.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _replay import show

show(["w0:195", "w1:294"],
     note="Core ML _upsample_bilinear2d_aa: antialias ignored -> large value error (max|delta| up to 2.33)")
