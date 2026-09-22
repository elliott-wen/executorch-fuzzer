#!/usr/bin/env python3
"""Repro: MediaTek Neuron delegate CORRUPTS int64 outputs (MISMATCH, delegated, systemic).
Every delegated op whose output is int64 comes back wrong. Two signatures, one root cause (int64 is
handled as/truncated to 32-bit lanes):
  * pure creation ops (fill/full/arange): values become 2^32-scale sentinels
    (4294967296 = 1<<32, 12884901888 = 3<<32) instead of the small ints eager produced.
  * shape/copy ops (squeeze/expand/constant_pad/permute/pixel_(un)shuffle): correct values but the
    trailing element(s) are zeroed (one int64 slot short — an int64/int32 element-count miscalc).
Mechanism: WRONG-VALUE / WRONG-DTYPE (int64 representation). Bucket: delegated. Deterministic 5/5.
The delegate should reject int64 ops rather than silently mangle them.
Run: broker up + phone worker (client-port 15555). `python repro_int64_corruption.py`"""
from _replay import show
show("w0:337",   "fill.Scalar int64 -> 2^32 sentinels")
show("w105:212", "arange.out int64 -> 2^32 sentinels")
show("w102:169", "full.out int64 -> 2^32 sentinel")
show("w0:270",   "squeeze_copy.dim int64 -> zeroed tail")
show("w103:700", "constant_pad_nd int64 -> zeroed tail")
