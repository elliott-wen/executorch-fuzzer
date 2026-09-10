#!/usr/bin/env python3
"""Buggy model graph (ARM XNNPACK): NEON tan reduced-precision. out[3] (float16) eager=[-0.300537] vs DEVICE(ARM)=[0.0] (max|delta|=0.30).
The erf instance is the analogous corpus/xnnpack/w10/w10_1362.py (out[7] -0.842773 -> 0.0).

This is the representative corpus graph w12:1473 (single-op minimisation does not delegate;
the bug needs the partitioned subgraph). Running it lowers to XNNPACK, dispatches to the
connected ARM phone, and prints device vs eager.

    .venv/bin/python findings/xnnpack-arm/bugs/graph_neon-tan-erf-precision.py
Prereq: `python broker.py` running + the ARM phone attached.
"""
import torch
torch.manual_seed(12648430)  # reproducible leaf data

def _make(shape, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype)
    lo = -4 if dtype.is_signed else 0  # unsigned dtypes reject a negative low
    return torch.randint(lo, 5, shape, dtype=dtype)

def _first(r):
    if isinstance(r, torch.Tensor): return r
    if isinstance(r, (tuple, list)):
        for x in r:
            if isinstance(x, torch.Tensor): return x
    return r

L0 = _make((4,), torch.int64)
L1 = _make((1,), torch.int64)
L2 = _make((), torch.int64)

def g(L0, L1, L2):
    n0 = _first(torch.ops.aten.div.out_mode(L0, L1, rounding_mode='trunc', out=torch.empty((4,), dtype=torch.int64)))
    n1 = _first(torch.ops.aten.bitwise_right_shift.Tensor_Scalar(n0, -4))
    n2 = _first(torch.ops.aten.sub.Scalar(n1, -4, -4))
    n3 = _first(torch.ops.aten.hardtanh.out(n2, 3, 3, out=n0))
    n4 = _first(torch.ops.aten.slice_copy.Tensor(n2, 0, 0, 1, 1))
    n5 = _first(torch.ops.aten.logical_or.out(n4, n3, out=torch.empty((4,), dtype=torch.int64)))
    n6 = _first(torch.ops.aten.erf.out(n0, out=torch.empty((4,), dtype=torch.float16)))
    n7 = _first(torch.ops.aten.cumsum.out(n5, 0, dtype=None, out=torch.empty((4,), dtype=torch.float32)))
    n8 = _first(torch.ops.aten.select_scatter.default(n3, L2, 0, 3))
    n9 = _first(torch.ops.aten.view_copy.default(n3, [4]))
    n10 = _first(torch.ops.aten.narrow_copy.out(n6, 0, 0, 0, out=torch.empty((0,), dtype=torch.float16)))
    n11 = _first(torch.ops.aten.max.default(n1))
    n12 = _first(torch.ops.aten.le.Scalar_out(n6, -1, out=n2))
    n13 = _first(torch.ops.aten.tan.out(n4, out=torch.empty((1,), dtype=torch.float16)))
    n14 = _first(torch.ops.aten.amin.out(n12, [], False, out=torch.empty((), dtype=torch.int64)))
    n15 = _first(torch.ops.aten.neg.out(n11, out=torch.empty((), dtype=torch.int64)))
    n16 = _first(torch.ops.aten.logical_not.out(n8, out=torch.empty((4,), dtype=torch.float32)))
    return (n7, n9, n10, n13, n14, n15, n16,)

LEAVES = [L0, L1, L2]
_BACKEND = 'inductor'

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/data/jwen929/mobile/findings/xnnpack-arm/bugs")
    from _trigger import trigger
    sys.exit(trigger(__file__, "min:tan", "w12:1473"))
