#!/usr/bin/env python3
"""Repro: `native_group_norm` produces WRONG normalized VALUES on the MediaTek delegate (MISMATCH).
fp32, finite input and finite eager reference; the device normalization diverges well beyond fp16
tolerance (max|delta| ~0.4-1.0 on unit-scale outputs). 97 value-mismatch samples; deterministic 5/5.
Failure mode: MISMATCH. Mechanism: WRONG-VALUE (group statistics / affine in the op's lowering).
Bucket: delegated. NB: group_norm ALSO has 176 nonfinite 'mismatches' that are RULED OUT — those are
a degenerate all-NaN eager reference, not a device bug (see ../ruled_out/nonfinite-reference-confound.md).
Run: broker up + phone worker (client-port 15555). `python repro_native_group_norm.py`"""
from _replay import show
show("w1:114", "native_group_norm -> wrong normalized values")
