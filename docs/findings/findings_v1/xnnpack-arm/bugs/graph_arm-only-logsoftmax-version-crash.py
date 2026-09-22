#!/usr/bin/env python3
"""Buggy model graph (ARM XNNPACK): ARM-only CRASH (version confound). x86 SKIPs (op_log_softmax.cpp:152 Check(false)); the AAR-source
implements _log_softmax so ARM runs on to the SHARED narrow_copy/unfold native abort. Chain: _log_softmax -> narrow_copy.

This is the representative corpus graph w0:480 (single-op minimisation does not delegate;
the bug needs the partitioned subgraph). Running it lowers to XNNPACK, dispatches to the
connected ARM phone, and prints device vs eager.

    .venv/bin/python findings/xnnpack-arm/bugs/graph_arm-only-logsoftmax-version-crash.py
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

L0 = _make((2,), torch.float16)
L1 = _make((), torch.int64)
L2 = _make((4,), torch.int64)

def g(L0, L1, L2):
    n0 = _first(torch.ops.aten._log_softmax.out(L0, 0, False, out=torch.empty((2,), dtype=torch.float16)))
    n1 = _first(torch.ops.aten.atan2.out(n0, n0, out=torch.empty((2,), dtype=torch.float32)))
    n2 = _first(torch.ops.aten.narrow_copy.out(n0, 0, 0, 0, out=torch.empty((0,), dtype=torch.float16)))
    n3 = _first(torch.ops.aten.native_layer_norm.default(n2, [0], None, None, 0.0))
    n4 = _first(torch.ops.aten.neg.out(n1, out=torch.empty((2,), dtype=torch.float32)))
    n5 = _first(torch.ops.aten.repeat.default(n4, [2]))
    n6 = _first(torch.ops.aten.broadcast_to.default(n5, [2, 4]))
    n7 = _first(torch.ops.aten._native_batch_norm_legit_no_training.default(n6, None, None, n5, n5, 0.0, 0.0))
    n8 = _first(torch.ops.aten.log2.out(n5, out=torch.empty((4,), dtype=torch.float16)))
    n9 = _first(torch.ops.aten.ne.Scalar_out(n7, -5, out=torch.empty((2, 4), dtype=torch.int64)))
    n10 = _first(torch.ops.aten.broadcast_to.default(n5, [2, 4, 4]))
    n11 = _first(torch.ops.aten.div.out_mode(n5, n10, rounding_mode=None, out=torch.empty((2, 4, 4), dtype=torch.float32)))
    n12 = _first(torch.ops.aten.alias_copy.default(n5))
    n13 = _first(torch.ops.aten.narrow_copy.default(n1, -1, 0, 0))
    n14 = _first(torch.ops.aten._to_copy.default(n4, dtype=torch.int64))
    n15 = _first(torch.ops.aten.slice_copy.Tensor(n14, 0, 0, 1, 1))
    n16 = _first(torch.ops.aten.ge.Tensor_out(n1, n15, out=torch.empty((2,), dtype=torch.int64)))
    n17 = _first(torch.ops.aten.floor_divide.default(L1, n13))
    n18 = _first(torch.ops.aten.narrow_copy.default(n12, 0, 0, 0))
    n19 = _first(torch.ops.aten.logical_and.out(n8, L2, out=torch.empty((4,), dtype=torch.int64)))
    return (n3, n9, n11, n16, n17, n18, n19,)

LEAVES = [L0, L1, L2]
_BACKEND = 'inductor'

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/data/jwen929/mobile/findings/xnnpack-arm/bugs")
    from _trigger import trigger
    sys.exit(trigger(__file__, "min:lsmax_crash", "w0:480"))
