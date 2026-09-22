#!/usr/bin/env python3
"""CONFIRMED bug (5/5): ENN integer-tensor data-movement is broken.
GARBAGE READ  — permute_copy/roll/t_copy(int) emit ~2^16-scale garbage unrelated to input.
DROPPED STORE — constant_pad_nd/expand_copy/squeeze_copy(int) zero the trailing elements.
Mechanism: WRONG-VALUE / ZEROED (delegated, ops>=1). Root: wrong integer element width/stride
in the ENN copy/layout path. Replays the exact corpus .pte on the Exynos E9965 device.
"""
from _replay import replay
replay([
    "w0:659",    # permute_copy  int32  -> [-131075, 327677, ...]   (garbage read)
    "w92:229",   # roll          int32  -> [-131075, -196611, ...]  (garbage read)
    "w18:507",   # t_copy        int64  -> [-131075, -3, -1, ...]   (garbage read)
    "w16:603",   # constant_pad_nd int64 -> trailing [-4,-2] become [0,0]  (dropped store)
    "w104:620",  # expand_copy   int32  -> trailing [-4] becomes [0]       (dropped store)
    "w103:113",  # squeeze_copy.dim int32 -> trailing zeroed
])
