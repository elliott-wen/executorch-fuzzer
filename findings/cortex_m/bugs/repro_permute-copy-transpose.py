#!/usr/bin/env python3
"""Repro for cortex-m permute-copy → cortex_m::transpose WRONG-VALUE bug.
See _repro_common.py for prerequisites (broker + FVP cortex-m client on 15664/15666).

The graph is a single `permute_copy(L0, [0,1])` — the IDENTITY permutation, so the correct
output is exactly the input. The CortexM pass rewrites permute_copy into `cortex_m::transpose`,
which unconditionally swaps axes, so the device returns the transposed (scrambled) tensor
instead of the identity. Deterministic REPRO 5/5 on the Corstone FVP; finite inputs; the
op-under-test ran as a cortex_m:: compute kernel (CM-COMPUTE).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _repro_common import run

JOB_ID = "w116:752"          # permute_copy(L0, [0,1]) on a (4,3) fp32 leaf; identity perm
# other confirmed instances (all identity perms, REPRO 5/5):
#   w7:164 [0,1] (4,3) | w14:781 [0,1,2,3] | w68:501 [0,1] | w27:220 [0,1,2,3] (4,2,4,3)
#   w78:676 | w61:597

if __name__ == "__main__":
    run(JOB_ID)
