#!/usr/bin/env python3
"""repro_avg_pool2d.py — Core ML: avg_pool2d wrong divisor (~2x) (WRONG-VALUE).

Run: PYTHONPATH=/data/jwen929 .venv/bin/python findings/coreml_mac/bugs/repro_avg_pool2d.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _replay import show

show(["w104:770", "w117:97"],
     note="Core ML avg_pool2d: pooled average off by ~2x (wrong divisor / count_include_pad)")
