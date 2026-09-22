#!/usr/bin/env python3
"""repro_precision-reductions.py — Core ML: reductions/activations beyond tolerance (precision).

Run: PYTHONPATH=/data/jwen929 .venv/bin/python findings/coreml_mac/bugs/repro_precision-reductions.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _replay import show

show(["w1:284", "w0:93", "w0:20", "w0:649", "w10:755", "w74:40", "w100:93"],
     note="Core ML precision losses (var/batch_norm/logit/pow/remainder/conv) beyond rtol=0.01")
