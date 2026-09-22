#!/usr/bin/env python3
"""Repro: `aten.roll` computes WRONG VALUES on the MediaTek Neuron delegate (MISMATCH, delegated).
roll should cyclically shift elements; the device instead emits a wrong ordering and fills the tail
with a repeated element (see device vs eager below). 187 value-mismatch samples; deterministic 5/5.
Failure mode: MISMATCH. Mechanism: WRONG-VALUE (data-movement / shift-index bug). Bucket: delegated.
Affects fp16 and fp32. Run: broker up + phone worker (client-port 15555). `python repro_roll.py`"""
from _replay import show
show("w0:232", "aten.roll -> wrong element ordering")
