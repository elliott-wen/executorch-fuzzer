#!/usr/bin/env python3
"""Repro: `convolution` emits values in the WRONG LAYOUT on the MediaTek delegate (MISMATCH).
fp32; the device output contains roughly the right magnitudes but in wrong spatial positions (a
layout/stride confusion), plus a leading 0.0 — max|delta| up to ~5.6. 9 value-mismatch samples;
deterministic 5/5. Failure mode: MISMATCH. Mechanism: WRONG-VALUE (output layout). Bucket: delegated.
Run: broker up + phone worker (client-port 15555). `python repro_convolution.py`"""
from _replay import show
show("w119:243", "convolution -> wrong output layout")
