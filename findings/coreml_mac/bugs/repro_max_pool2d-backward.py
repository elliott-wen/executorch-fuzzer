#!/usr/bin/env python3
"""repro_max_pool2d-backward.py — Core ML: max_pool2d backward wrong gradient scatter.

Run: PYTHONPATH=/data/jwen929 .venv/bin/python findings/coreml_mac/bugs/repro_max_pool2d-backward.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _replay import show

show(["w21:720", "w3:391"],
     note="Core ML max_pool2d_with_indices_backward: gradient scattered to wrong positions (max|delta|~2.4)")
