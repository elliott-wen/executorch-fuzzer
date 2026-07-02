"""runnable_ops.py — the op set for the mobile differential path.

Op set = the mobile executorch_allowlist (OVERLOAD-EXACT, regenerated from the
installed ExecuTorch runtime registry — an op is in it iff its .out kernel is
registered) minus the blocklist.

Because the allowlist is now overload-exact, ops the runtime can't run aren't in it
in the first place — no missing-kernel denylist is needed (the earlier base-name
allowlist over-included overloads like bitwise_*.Scalar_Tensor and native_batch_norm,
which is why a denylist used to exist).

Exclusions (unsafe primitives AND RNG ops) all live in blocklist.py and are applied by
load_opnodes(); this module just selects the ExecuTorch-runnable slice.
"""

from __future__ import annotations


def load_runnable_opnodes(verbose: bool = True):
    """OpNodes for the full ExecuTorch allowlist minus the blocklist (which now also
    excludes RNG ops — eager vs ExecuTorch diverge on unshared randomness)."""
    from mobile.gen.ops.opnode import load_opnodes
    return load_opnodes(executorch=True)
