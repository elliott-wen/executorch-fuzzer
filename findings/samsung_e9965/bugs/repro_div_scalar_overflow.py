#!/usr/bin/env python3
"""CONFIRMED bug (5/5): div.Scalar overflows fp16 on ENN from a FINITE input.
eager (fp32) [-0.482, 1.110, 0.934] -> device [-31600, inf, 61184]. Dividing by a small scalar
overflows the fp16 NPU range while the reference stays finite. Mechanism: NONFINITE (delegated).
Distinct from the ruled-out nan/inf-INPUT cases (this leaf is finite). Replays exact corpus .pte.
"""
from _replay import replay
replay(["w0:297"])
