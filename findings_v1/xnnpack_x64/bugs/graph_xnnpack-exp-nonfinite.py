#!/usr/bin/env python3
"""Buggy model graph (x86 XNNPACK): exp NaN/overflow. The configured host exp kernel is the
rational-polynomial approximation (xnn_f32_vexp_ukernel__{sse2,avx,fma3,avx512f}_rational_3_2_div),
NOT libm expf. Its range reduction uses integer bit-manipulation ((x+magic)<<23) with no NaN/Inf
guard, so exp(NaN) -> 0 (and large-overflow -> garbage finite). Eager: exp(NaN)=NaN, exp(+big)=+inf.

Representative corpus graph w0:938. Localizer-confirmed born-here:
    corpus/xnnpack/w0/w0_938.py -> Divergence(node='n12', op='exp', kind='nonfinite', index=12) e[0]=nan b[0]=0

    .venv/bin/python findings/xnnpack_x64/bugs/graph_xnnpack-exp-nonfinite.py
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

L0 = _make((4, 2), torch.int64)

def g(L0):
    n0 = _first(torch.ops.aten.sum.IntList_out(L0, None, False, dtype=None, out=torch.empty((), dtype=torch.int64)))
    n1 = _first(torch.ops.aten.neg.out(n0, out=torch.empty((), dtype=torch.int64)))
    n2 = _first(torch.ops.aten.bitwise_xor.Scalar(n0, 6))
    n3 = _first(torch.ops.aten.isnan.default(n0))
    n4 = _first(torch.ops.aten.asin.out(n3, out=torch.empty((), dtype=torch.float16)))
    n5 = _first(torch.ops.aten.clone.default(n0, memory_format=None))
    n6 = _first(torch.ops.aten.relu.default(n1))
    n7 = _first(torch.ops.aten.min.unary_out(n3, out=torch.empty((), dtype=torch.int64)))
    n8 = _first(torch.ops.aten.acosh.out(n4, out=torch.empty((), dtype=torch.float16)))
    n9 = _first(torch.ops.aten.tan.out(n4, out=torch.empty((), dtype=torch.float16)))
    n10 = _first(torch.ops.aten.sum.IntList_out(n8, None, False, dtype=None, out=torch.empty((), dtype=torch.int64)))
    n11 = _first(torch.ops.aten.roll.default(n5, [-3], []))
    n12 = _first(torch.ops.aten.exp.out(n8, out=torch.empty((), dtype=torch.float16)))
    n13 = _first(torch.ops.aten.asinh.out(n4, out=torch.empty((), dtype=torch.float16)))
    n14 = _first(torch.ops.aten.rsub.Scalar(n10, -1.0, 6.0))
    n15 = _first(torch.ops.aten.broadcast_to.default(n14, [2, 2, 2, 1, 2]))
    n16 = _first(torch.ops.aten.gt.Tensor_out(n15, n0, out=torch.empty((2, 2, 2, 1, 2), dtype=torch.float16)))
    return (n2, n6, n7, n9, n11, n12, n13, n16,)

LEAVES = [L0]
_BACKEND = 'inductor'

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/data/jwen929/mobile/findings/xnnpack_x64/bugs")
    from _trigger import trigger
    sys.exit(trigger(__file__,
        "Divergence(node='n12', op='exp', kind='nonfinite') e[0]=nan b[0]=0"))
