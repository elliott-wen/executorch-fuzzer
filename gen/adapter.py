"""adapter.py — natural, fusion-transparent glue for connecting a producer's
output into an op's tensor port.

Policy (see design discussion): connect a producer to a port via adapters the
ExecuTorch backends treat transparently (verified against exir/XNNPACK passes) —
  - dtype cast  (_to_copy):     fused/removed (RemoveToCopyPass; inductor too)
  - broadcast   (broadcast_to): stride-0 view, native to pointwise fusion
  - reshape     (view_copy):    a ZERO-COPY ALIAS in exir memory planning
        (ReplaceViewCopyWithViewPass → memory.view shares the base's mem_id/
        mem_offset) and a delegated op in XNNPACK (ViewCopyConfig → static
        reshape). It preserves the element COUNT exactly (numel(src)==numel(dst))
        — it only rearranges shape, never adds/drops elements.
We still avoid element-COUNT-changing glue (pad adds fabricated zeros; repeat/
tile duplicate into new storage), so when no cast/broadcast/reshape fits, the
caller uses a fresh leaf instead.
"""

from __future__ import annotations

import math

import torch


class AdapterNode:
    """Minimal node mimicking the OpNode interface that GenGraph needs
    (op / op_name / named_params / label), for inserted glue ops."""

    def __init__(self, op, op_name, named_params, label):
        self.op = op
        self.op_name = op_name
        self.named_params = named_params
        self.label = label


def make_cast() -> AdapterNode:
    return AdapterNode(torch.ops.aten._to_copy.default, "_to_copy",
                       [("self", "Tensor"), ("dtype", "int?")], "to")


def make_broadcast() -> AdapterNode:
    return AdapterNode(torch.ops.aten.broadcast_to.default, "broadcast_to",
                       [("self", "Tensor"), ("size", "int[]")], "bcast")


def make_reshape() -> AdapterNode:
    """Shape adapter: rearrange a producer into a port's shape WITHOUT changing the
    element count (numel preserved). view_copy is a zero-copy alias in exir memory
    planning and a delegated static-reshape in XNNPACK."""
    return AdapterNode(torch.ops.aten.view_copy.default, "view_copy",
                       [("self", "Tensor"), ("size", "int[]")], "rshp")


def make_slice() -> AdapterNode:
    """Downsize adapter: take a stride-1 sub-region of ONE dim (chained per dim to
    shrink a bigger producer into a smaller port). slice_copy preserves real data
    and is a delegated XNNStaticSlice in XNNPACK (constraint: step==1, static)."""
    return AdapterNode(torch.ops.aten.slice_copy.Tensor, "slice_copy.Tensor",
                       [("self", "Tensor"), ("dim", "int"), ("start", "int"),
                        ("end", "int"), ("step", "int")], "slice")


def make_pad() -> AdapterNode:
    """Upsize adapter: zero-pad the tail of each dim to grow a smaller producer into
    a larger port. constant_pad_nd is a delegated XNNStaticConstantPad in XNNPACK
    (constraint: non-negative pads). NOTE: fabricates zeros (changes values)."""
    return AdapterNode(torch.ops.aten.constant_pad_nd.default, "constant_pad_nd",
                       [("self", "Tensor"), ("pad", "int[]"), ("value", "Scalar")],
                       "pad")


def make_clamp() -> AdapterNode:
    """Range adapter: clamp a producer's values into a port's required [lo, hi].
    Used when wiring a producer into a value-constrained port (index/target/...),
    whose data range a plain shape/dtype pin can't enforce. Pointwise → fuses."""
    return AdapterNode(torch.ops.aten.clamp.default, "clamp",
                       [("self", "Tensor"), ("min", "Scalar?"), ("max", "Scalar?")],
                       "clampr")


def _broadcastable_to(src, dst) -> bool:
    """True iff a tensor of shape `src` broadcasts to shape `dst`."""
    if len(src) > len(dst):
        return False
    for s, d in zip(reversed(src), reversed(dst)):
        if s != d and s != 1:
            return False
    return True


def _numel(shape) -> int:
    return math.prod(shape) if len(shape) else 1


def plan_connection(src_shape, src_dtype, want_shape, want_dtype):
    """How to connect a producer (src_shape/dtype) into a port wanting
    (want_shape/dtype): 'direct' | 'cast' | 'broadcast' | 'broadcast_cast' |
    'reshape' | 'reshape_cast' | 'slice' | 'slice_cast' | 'pad' | 'pad_cast',
    or None when nothing fits (caller uses a leaf).

    Preference order is most-transparent / most-faithful first:
      identity > dtype-cast > stride-0 broadcast > numel-preserving reshape >
      real-data downsize (slice) > zero-fabricating upsize (pad).
    slice/pad require equal rank; slice shrinks every differing dim (want<=src),
    pad grows every differing dim (want>=src). All verified non-barrier in the
    XNNPACK partitioner (slice_copy / constant_pad_nd are delegated)."""
    src_shape = tuple(src_shape)
    want_shape = tuple(want_shape)
    same_shape = src_shape == want_shape
    bcast = _broadcastable_to(src_shape, want_shape)
    same_numel = _numel(src_shape) == _numel(want_shape)
    same_rank = len(src_shape) == len(want_shape)
    can_slice = same_rank and all(w <= s for w, s in zip(want_shape, src_shape))
    can_pad = same_rank and all(w >= s for w, s in zip(want_shape, src_shape))
    same_dtype = src_dtype == want_dtype
    suffix = "" if same_dtype else "_cast"
    if same_shape and same_dtype:
        return "direct"
    if same_shape:
        return "cast"
    if bcast:
        return "broadcast" + suffix
    if same_numel:
        return "reshape" + suffix
    if can_slice:
        return "slice" + suffix
    if can_pad:
        return "pad" + suffix
    return None
