#!/usr/bin/env python3
"""Buggy model graph (x86 XNNPACK): sqrt/rsqrt negative-domain. XNNPACK computes
sqrt(x)=x*rsqrt(x) via the fast reciprocal-sqrt approximation (x86 _mm_rsqrt_ps + Newton),
whose special-case mask handles only +-0, so sqrt(negative) -> -0 instead of NaN. The input
is a genuine negative real, not a NaN, so this is a true domain divergence (distinct from the
clamp/min/max NaN-laundering family). Eager: sqrt(x<0)=NaN.

Representative corpus graph w0:963. Localizer-confirmed born-here:
    corpus/xnnpack/w0/w0_963.py -> Divergence(node='n0', op='sqrt', kind='nonfinite', index=0)
    n0 input [-0.482, 1.110] eager [nan, 1.054] xnnpack [-0, ...]

    .venv/bin/python findings/xnnpack_x64/bugs/graph_xnnpack-sqrt-rsqrt-domain.py
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

L0 = _make((2,), torch.float16)
L1 = _make((2,), torch.int64)
L2 = _make((), torch.int64)

def g(L0, L1, L2):
    n0 = _first(torch.ops.aten.sqrt.out(L0, out=torch.empty((2,), dtype=torch.float16)))
    n1 = _first(torch.ops.aten.broadcast_to.default(n0, [3, 2]))
    n2 = _first(torch.ops.aten._native_batch_norm_legit_no_training.default(n1, None, None, n0, n0, 0.0, 0.0))
    n3 = _first(torch.ops.aten.atan.out(n1, out=torch.empty((3, 2), dtype=torch.float32)))
    n4 = _first(torch.ops.aten.asinh.out(n3, out=torch.empty((3, 2), dtype=torch.float32)))
    n5 = _first(torch.ops.aten.sign.out(n3, out=torch.empty((3, 2), dtype=torch.float32)))
    n6 = _first(torch.ops.aten.round.out(n0, out=torch.empty((2,), dtype=torch.float16)))
    n7 = _first(torch.ops.aten.constant_pad_nd.default(n0, [], 4))
    n8 = _first(torch.ops.aten.div.Scalar(n4, 8))
    n9 = _first(torch.ops.aten.div.Scalar(n8, -5))
    n10 = _first(torch.ops.aten.prod.default(n9, dtype=None))
    n11 = _first(torch.ops.aten.rsub.Scalar(n2, 1, 7.0))
    n12 = _first(torch.ops.aten.div.out(n10, n10, out=torch.empty((), dtype=torch.float16)))
    n13 = _first(torch.ops.aten.logit.out(n10, None, out=torch.empty((), dtype=torch.float16)))
    n14 = _first(torch.ops.aten.logical_xor.default(L1, n2))
    n15 = _first(torch.ops.aten.le.Tensor_out(n12, L2, out=torch.empty((), dtype=torch.int64)))
    n16 = _first(torch.ops.aten.broadcast_to.default(n15, [2]))
    n17 = _first(torch.ops.aten.clamp.default(n16, 0, 0))
    n18 = _first(torch.ops.aten.index_select.default(n7, 0, n17))
    return (n5, n6, n11, n13, n14, n18,)

LEAVES = [L0, L1, L2]
_BACKEND = 'inductor'

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/data/jwen929/mobile/findings/xnnpack_x64/bugs")
    from _trigger import trigger
    sys.exit(trigger(__file__,
        "Divergence(node='n0', op='sqrt', kind='nonfinite') eager=[nan,1.054] xnnpack=[-0,...]"))
