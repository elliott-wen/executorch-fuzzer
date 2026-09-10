"""adapter.py — the glue that connects a producer's output into a consumer's port.

Two shapes rarely line up by chance, so most wiring needs a converter in between. Which
converter is not a free choice: the graph exists to test a backend, so the glue must be
something backends treat transparently, or the glue becomes the thing under test.

Verified against exir / the XNNPACK partitioner, most faithful first:
  cast        `_to_copy`         removed outright (RemoveToCopyPass)
  broadcast   `broadcast_to`     a stride-0 view, native to pointwise fusion
  reshape     `view_copy`        a zero-copy alias in exir memory planning
                                 (ReplaceViewCopyWithViewPass), delegated by XNNPACK
  slice       `slice_copy`       real sub-region, delegated (XNNStaticSlice, step 1)
  pad         `constant_pad_nd`  delegated (XNNStaticConstantPad) — but FABRICATES zeros
  clamp       `clamp`            pointwise, fuses; the only value-changing adapter, used
                                 where a port constrains its data range

cast/broadcast/reshape preserve the data exactly. slice keeps a real sub-region; pad
invents zeros. When nothing fits, the caller uses a fresh leaf instead of forcing glue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Adapter:
    """A glue node. Carries the same surface the graph and emitter read off a real Op
    (`fn`, `op_name`, `params`, `label`) and nothing else — an adapter is chosen, never
    solved for, so it needs no constraints."""

    fn: Any
    op_name: str
    params: list[tuple[str, str]]
    label: str


def make_cast() -> Adapter:
    return Adapter(torch.ops.aten._to_copy.default, "_to_copy",
                   [("self", "Tensor"), ("dtype", "int?")], "to")


def make_broadcast() -> Adapter:
    return Adapter(torch.ops.aten.broadcast_to.default, "broadcast_to",
                   [("self", "Tensor"), ("size", "int[]")], "bcast")


def make_reshape() -> Adapter:
    """Rearrange into the port's shape without changing the element count."""
    return Adapter(torch.ops.aten.view_copy.default, "view_copy",
                   [("self", "Tensor"), ("size", "int[]")], "rshp")


def make_slice() -> Adapter:
    """Shrink one dim to a stride-1 sub-region; chained per dim that must shrink."""
    return Adapter(torch.ops.aten.slice_copy.Tensor, "slice_copy.Tensor",
                   [("self", "Tensor"), ("dim", "int"), ("start", "int"),
                    ("end", "int"), ("step", "int")], "slice")


def make_pad() -> Adapter:
    """Grow each dim by zero-padding its tail. Fabricates values — last resort."""
    return Adapter(torch.ops.aten.constant_pad_nd.default, "constant_pad_nd",
                   [("self", "Tensor"), ("pad", "int[]"), ("value", "Scalar")], "pad")


def make_clamp() -> Adapter:
    """Force a producer's values into a port's required [lo, hi]. Needed because a
    shape/dtype pin says nothing about DATA, and ports like index/target constrain it."""
    return Adapter(torch.ops.aten.clamp.default, "clamp",
                   [("self", "Tensor"), ("min", "Scalar?"), ("max", "Scalar?")], "clampr")


def _broadcasts_to(src: tuple[int, ...], dst: tuple[int, ...]) -> bool:
    if len(src) > len(dst):
        return False
    return all(s == d or s == 1 for s, d in zip(reversed(src), reversed(dst)))


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape) if shape else 1


def plan_connection(src_shape, src_dtype, want_shape, want_dtype) -> str | None:
    """Name the conversion that connects a producer into a port, or None if none fits.

    Preference runs most-faithful first: identity, then a dtype cast, then a stride-0
    broadcast, then a count-preserving reshape, then a real-data downsize, and only last a
    zero-fabricating upsize. slice and pad both need equal rank — slice when every
    differing dim shrinks, pad when every differing dim grows.
    """
    src_shape, want_shape = tuple(src_shape), tuple(want_shape)
    same_rank = len(src_shape) == len(want_shape)
    suffix = "" if src_dtype == want_dtype else "_cast"

    if src_shape == want_shape:
        return "direct" if not suffix else "cast"
    if _broadcasts_to(src_shape, want_shape):
        return "broadcast" + suffix
    if _numel(src_shape) == _numel(want_shape):
        return "reshape" + suffix
    if same_rank and all(w <= s for w, s in zip(want_shape, src_shape)):
        return "slice" + suffix
    if same_rank and all(w >= s for w, s in zip(want_shape, src_shape)):
        return "pad" + suffix
    return None
