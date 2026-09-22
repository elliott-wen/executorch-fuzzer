#!/usr/bin/env python3
"""Repro: `aten.copy` native-aborts the MediaTek Neuron executor (CRASH, delegated ops=1).
The one-op graph `copy(L0,L1,non_blocking)` (here (2,4) <- broadcast (1,4), fp32) kills the phone
executor process with a native abort — no catchable error. 217/217 copy jobs in the corpus crash;
reproduces 5/5 in isolation. Failure mode: CRASH. Bucket: delegated. Trigger: any copy form.
Native abort => no `Check failed` guard; the delegate should reject the op, not abort.
Run: broker up + phone worker connected (client-port 15555). `python repro_copy.py`"""
from _replay import show
show("w0:264", "aten.copy -> native abort")
