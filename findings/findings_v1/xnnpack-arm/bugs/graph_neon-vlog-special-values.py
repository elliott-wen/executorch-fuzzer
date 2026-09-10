#!/usr/bin/env python3
"""Minimal buggy model graph: NEON `f32-vlog` drops Inf/NaN.

`log(+Inf)` on the ARM XNNPACK NEON `f32-vlog` microkernel returns the finite constant
~88.376266 instead of `+Inf` (the x86 / scalar path returns `+Inf` correctly). One op.
`logit(x) = log(x/(1-x))` decomposes onto this same `log`, which is why the corpus's logit
cluster (~2,385 rows) is all this bug.

Run it (lowers to XNNPACK + feeds the connected ARM phone, prints device vs eager):
    cd /data/jwen929/mobile && .venv/bin/python findings/xnnpack-arm/bugs/graph_neon-vlog-special-values.py
Verified: eager=[Infinity,…]  DEVICE(ARM)=[88.376266,…]
"""
import torch


def _first(r):
    return r[0] if isinstance(r, (tuple, list)) else r


L0 = torch.full((4,), float("inf"), dtype=torch.float32)   # the runtime input: +Inf


def g(L0):
    return (_first(torch.ops.aten.log.default(L0)),)        # log(+Inf) → 88.376266 on ARM NEON


LEAVES = [L0]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/data/jwen929/mobile/findings/xnnpack-arm/bugs")
    from _trigger import trigger
    sys.exit(trigger(__file__, "min:vlog", "n0=log(L0)"))
