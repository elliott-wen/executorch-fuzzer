#!/usr/bin/env python3
"""Buggy model graph (x86 XNNPACK): .pte fails to LOAD (partition over-inclusion of >6-D
tensors). pixel_shuffle's decomposition produces a rank-7 view_copy+permute; the XNNPACK
partitioner has no XNN_MAX_TENSOR_DIMS(=6) rank guard, so it delegates the rank-7 node and the
serialized subgraph is rejected at load time by xnn_define_tensor_value /
xnn_define_static_transpose -> XNNCompiler::compileModel fails -> load_method fails, before any
execution. Eager/portable load and run fine.

Representative corpus graph w0:1175 (pixel_shuffle on a 5-D fp16 input -> rank-7 reshape).
Documented runtime error:
    [XNNCompiler.cpp:664] Failed to define tensor 0 with code: xnn_status_unsupported_parameter
    RuntimeError: Failed to load method forward, error: 0x:1

Unlike the other three bugs (which run and silently launder non-finite values), this one cannot
even load, so the trigger surfaces it as a localizer ERROR rather than a Divergence.

    .venv/bin/python findings/xnnpack_x64/bugs/graph_xnnpack-load-failure.py
See _trigger.py.
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

L0 = _make((4, 2, 4, 4, 2), torch.float16)

def g(L0):
    n0 = _first(torch.ops.aten.native_group_norm.default(L0, None, None, 2, 128, 1, 1, 0.0))
    n1 = _first(torch.ops.aten.native_layer_norm.default(n0, [2], None, None, 0.0))
    n2 = _first(torch.ops.aten.pixel_shuffle.default(n0, 2))
    n3 = _first(torch.ops.aten.argmin.out(n1, None, False, out=torch.empty((), dtype=torch.int64)))
    n4 = _first(torch.ops.aten.pow.Tensor_Scalar_out(n1, -7, out=torch.empty((4, 2, 4, 4, 2), dtype=torch.float32)))
    n5 = _first(torch.ops.aten.view_copy.default(n3, []))
    n6 = _first(torch.ops.aten.roll.default(n1, [-5, -3], [0, 2]))
    n7 = _first(torch.ops.aten.floor.out(n1, out=torch.empty((4, 2, 4, 4, 2), dtype=torch.float16)))
    n8 = _first(torch.ops.aten.transpose_copy.int(n3, 0, 0))
    n9 = _first(torch.ops.aten.round.out(n6, out=torch.empty((4, 2, 4, 4, 2), dtype=torch.float16)))
    n10 = _first(torch.ops.aten.hardtanh.out(n5, -3, 6, out=torch.empty((), dtype=torch.int64)))
    n11 = _first(torch.ops.aten.log10.out(n9, out=torch.empty((4, 2, 4, 4, 2), dtype=torch.float32)))
    n12 = _first(torch.ops.aten._conj_physical.default(n6))
    n13 = _first(torch.ops.aten._to_copy.default(n5, dtype=torch.float32))
    n14 = _first(torch.ops.aten.broadcast_to.default(n13, [2]))
    n15 = _first(torch.ops.aten.logical_and.default(n11, n14))
    n16 = _first(torch.ops.aten.slice_copy.Tensor(n7, 2, None, None, 6))
    n17 = _first(torch.ops.aten.ceil.out(n6, out=torch.empty((4, 2, 4, 4, 2), dtype=torch.float16)))
    return (n2, n4, n8, n10, n12, n15, n16, n17,)

LEAVES = [L0]
_BACKEND = 'inductor'

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/data/jwen929/mobile/findings/xnnpack_x64/bugs")
    from _trigger import trigger
    sys.exit(trigger(__file__,
        "ERROR / Failed to load method forward (xnn_status_unsupported_parameter; rank-7 tensor)"))
