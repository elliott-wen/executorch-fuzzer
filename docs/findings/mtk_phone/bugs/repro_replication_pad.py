#!/usr/bin/env python3
"""Repro: `replication_pad{1,2,3}d.out` on the MediaTek delegate — mostly SKIP, and WRONG-VALUE when
it does run. Two device behaviours, both delegated (partitioned into the Neuron delegate):
  * SKIP: most forms are rejected at RUNTIME with
      'ExecutorchInvalidArgumentException: [ExecuTorch Error 0x12] Invalid argument' — partitioned but
      not executable (a real coverage gap). 795 SKIP samples across 1d/2d/3d.
  * MISMATCH: the forms that DO execute produce a wrong element ordering (like roll). 11 samples.
Failure modes: SKIP (verbatim reason above) + MISMATCH (WRONG-VALUE). Bucket: delegated.
Run: broker up + phone worker (client-port 15555). `python repro_replication_pad.py`"""
from _replay import show
show("w0:259", "replication_pad1d SKIP (Invalid argument)")
show("w104:21", "replication_pad1d -> wrong ordering (executing form)")
