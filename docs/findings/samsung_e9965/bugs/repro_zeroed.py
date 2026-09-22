#!/usr/bin/env python3
"""CONFIRMED bug (5/5): ENN drops the store (all-zero output) where the reference is non-zero.
embedding / mul.Scalar(fp16) / pixel_unshuffle return all-zeros for finite non-zero references.
Mechanism: ZEROED (delegated, ops>=1). Replays the exact corpus .pte on the Exynos E9965 device.
"""
from _replay import replay
replay([
    "w10:175",   # embedding      -> device all-zero
    "w1:355",    # mul.Scalar fp16 -> device all-zero
    "w112:9",    # pixel_unshuffle -> device all-zero
])
