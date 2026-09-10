#!/usr/bin/env python3
"""Buggy model graph (x86 XNNPACK): clamp/min/max NaN-laundering. The XNNPACK delegate lowers
relu/clamp/hardtanh -> XNNClamp and minimum/maximum -> XNNMinimum/XNNMaximum, whose SIMD
fmin/fmax (x86 _mm_min_ps/_mm_max_ps) return the NON-NaN operand, so a NaN is laundered to a
finite bound/operand. Eager propagates NaN.

Representative corpus graph w0:49. Localizer-confirmed born-here:
    corpus/xnnpack/w0/w0_49.py -> Divergence(node='n17', op='hardtanh', kind='nonfinite', index=17) e[0]=nan b[0]=-2
(the same graph also laundres minimum n9 and relu n10). Single-op isolation does not delegate;
the bug needs the partitioned subgraph, so this uses the full corpus graph.

    .venv/bin/python findings/xnnpack_x64/bugs/graph_xnnpack-clamp-minmax-nan.py
Reproduces in-process on the x86 host (no phone needed); see _trigger.py.
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

L0 = _make((2,), torch.float32)
L1 = _make((0,), torch.int64)
L2 = _make((0,), torch.float32)
L3 = _make((1,), torch.bool)
L4 = _make((2,), torch.float32)

def g(L0, L1, L2, L3, L4):
    n0 = _first(torch.ops.aten.scatter_add.out(L0, -1, L1, L2, out=torch.empty((2,), dtype=torch.float32)))
    n1 = _first(torch.ops.aten.atanh.out(n0, out=torch.empty((2,), dtype=torch.float16)))
    n2 = _first(torch.ops.aten.eq.Tensor_out(n1, n1, out=torch.empty((2,), dtype=torch.int64)))
    n3 = _first(torch.ops.aten.mean.dtype_out(n2, dtype=torch.float16, out=torch.empty((), dtype=torch.float16)))
    n4 = _first(torch.ops.aten.remainder.Tensor_out(n0, n0, out=torch.empty((2,), dtype=torch.float32)))
    n5 = _first(torch.ops.aten.acosh.out(n3, out=torch.empty((), dtype=torch.float16)))
    n6 = _first(torch.ops.aten.gelu.out(n3, approximate='tanh', out=torch.empty((), dtype=torch.float16)))
    n7 = _first(torch.ops.aten._to_copy.default(n6, dtype=torch.float32))
    n8 = _first(torch.ops.aten.broadcast_to.default(n7, [2]))
    n9 = _first(torch.ops.aten.minimum.out(n4, n8, out=torch.empty((2,), dtype=torch.float32)))
    n10 = _first(torch.ops.aten.relu.default(n9))
    n11 = _first(torch.ops.aten.sub.out(n7, n7, alpha=8, out=torch.empty((), dtype=torch.float16)))
    n12 = _first(torch.ops.aten.atan.out(n7, out=torch.empty((), dtype=torch.float16)))
    n13 = _first(torch.ops.aten.masked_scatter.default(n0, L3, L4))
    n14 = _first(torch.ops.aten._to_copy.default(n6, dtype=torch.int64))
    n15 = _first(torch.ops.aten.broadcast_to.default(n14, [0]))
    n16 = _first(torch.ops.aten.copy.default(n15, n7, False))
    n17 = _first(torch.ops.aten.hardtanh.out(n5, -2, 7, out=torch.empty((), dtype=torch.float16)))
    n18 = _first(torch.ops.aten.max.default(n7))
    n19 = _first(torch.ops.aten.rsqrt.out(n7, out=torch.empty((), dtype=torch.float16)))
    return (n10, n11, n12, n13, n16, n17, n18, n19,)

LEAVES = [L0, L1, L2, L3, L4]
_BACKEND = 'inductor'

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/data/jwen929/mobile/findings/xnnpack_x64/bugs")
    from _trigger import trigger
    sys.exit(trigger(__file__,
        "Divergence(node='n17', op='hardtanh', kind='nonfinite') e[0]=nan b[0]=-2"))
