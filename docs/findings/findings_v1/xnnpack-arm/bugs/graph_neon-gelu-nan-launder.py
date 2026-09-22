#!/usr/bin/env python3
"""Buggy model graph (ARM XNNPACK): NEON gelu launders NaN -> 0. Diverging output: out[4] (float16) eager=[1.657227, nan] vs DEVICE(ARM)=[1.657227, 0.0].

This is the representative corpus graph w0:1273 (single-op minimisation does not delegate;
the bug needs the partitioned subgraph). Running it lowers to XNNPACK, dispatches to the
connected ARM phone, and prints device vs eager.

    .venv/bin/python findings/xnnpack-arm/bugs/graph_neon-gelu-nan-launder.py
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

L0 = _make((2, 1, 4), torch.int64)

def g(L0):
    n0 = _first(torch.ops.aten.stack.default([torch.tensor([11, 0], dtype=torch.int64)], 1))
    n1 = _first(torch.ops.aten.acos.out(n0, out=torch.empty((2, 1), dtype=torch.float16)))
    n2 = _first(torch.ops.aten.lift_fresh_copy.default(n0))
    n3 = _first(torch.ops.aten.sqrt.out(n2, out=torch.empty((2, 1), dtype=torch.float16)))
    n4 = _first(torch.ops.aten.trunc.out(n0, out=torch.empty((2, 1), dtype=torch.int64)))
    n5 = _first(torch.ops.aten.log2.out(n3, out=torch.empty((2, 1), dtype=torch.float16)))
    n6 = _first(torch.ops.aten.unbind_copy.int(n5, 0))
    n7 = _first(torch.ops.aten.div.out_mode(n3, L0, rounding_mode=None, out=torch.empty((2, 2, 4), dtype=torch.float16)))
    n8 = _first(torch.ops.aten.narrow_copy.default(n5, 1, 0, 0))
    n9 = _first(torch.ops.aten.logical_not.default(n1))
    n10 = _first(torch.ops.aten._to_copy.default(n6, dtype=torch.int64))
    n11 = _first(torch.ops.aten.broadcast_to.default(n10, [0, 0]))
    n12 = _first(torch.ops.aten.clamp.default(n11, 0, 0))
    n13 = _first(torch.ops.aten.gather.out(n3, -2, n12, sparse_grad=False, out=torch.empty((0, 0), dtype=torch.float16)))
    n14 = _first(torch.ops.aten.gelu.out(n5, approximate='none', out=torch.empty((2, 1), dtype=torch.float16)))
    n15 = _first(torch.ops.aten.log2.out(n5, out=torch.empty((2, 1), dtype=torch.float16)))
    n16 = _first(torch.ops.aten.neg.out(n1, out=torch.empty((2, 1), dtype=torch.float16)))
    n17 = _first(torch.ops.aten.broadcast_to.default(n10, [2, 1, 4]))
    n18 = _first(torch.ops.aten.eq.Tensor_out(n0, n17, out=torch.empty((2, 2, 4), dtype=torch.int64)))
    n19 = _first(torch.ops.aten.sum.IntList_out(n13, None, False, dtype=None, out=torch.empty((), dtype=torch.int64)))
    return (n4, n7, n8, n9, n14, n15, n16, n18, n19,)

LEAVES = [L0]
_BACKEND = 'inductor'

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/data/jwen929/mobile/findings/xnnpack-arm/bugs")
    from _trigger import trigger
    sys.exit(trigger(__file__, "min:gelu", "w0:1273"))
