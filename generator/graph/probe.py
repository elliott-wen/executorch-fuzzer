"""probe.py — validate a candidate node and learn the shape it produces.

Every node is checked by running its op on META tensors: shape and dtype only, no data and
no kernel. That keeps generation cheap, and it keeps a broken kernel from taking the
generation worker down — a real crash belongs to the RUN phase, where it is the point.

Most of this file is torch-schema arcana rather than logic. Calling an arbitrary ATen
overload correctly means honouring kwarg-only arguments and synthesizing an `out=` the
model never built. Ops the runtime has no meta rule for are handled by supplying one —
see meta_impls.py.
"""

from __future__ import annotations

import torch

from mobile.generator.graph.meta_impls import register as _register_meta_impls

_register_meta_impls()   # ops the runtime has no meta rule for; see meta_impls.py



def _to_meta(x):
    """Move a value to the meta device, elementwise through lists.

    A `Tensor[]` port has to be converted element by element: leaving a list of CPU
    tensors alongside a synthesized meta `out=` makes the op fail on a device mismatch,
    and the probe would wrongly read that as an invalid node.
    """
    if isinstance(x, torch.Tensor):
        return x.to("meta")
    if isinstance(x, (list, tuple)):
        return type(x)(_to_meta(e) for e in x)
    return x


def _first_tensor(result):
    """The first tensor in an op's result, which may be a bare tensor or a tuple."""
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (tuple, list)):
        return next((x for x in result if isinstance(x, torch.Tensor)), None)
    return None


def _split_by_schema(fn, args: list) -> tuple[list, dict]:
    """Split a flat argument list into (positional, keyword) using the op's JIT schema.

    Arguments after `*` in the schema must be passed by keyword or the call raises. Two
    subtleties: a required kwarg-only argument has to be passed even when its value is
    None (`div.out_mode`'s `rounding_mode`), and a structured op's `out=` may be missing
    from the list entirely — meta-only ops reach it through maybe_get_output, so it is
    never built. The `.out` overload still demands it, so an empty tensor is synthesized
    (the op resizes it).
    """
    schema = getattr(fn, "_schema", None)
    if schema is None:
        return list(args), {}
    try:
        positional, keyword = [], []
        first_dtype = None
        for value, arg in zip(args, schema.arguments):
            if arg.kwarg_only:
                required = not arg.has_default_value() if hasattr(arg, "has_default_value") else False
                if value is not None or required:
                    keyword.append((arg.name, value))
            else:
                positional.append(value)
                if first_dtype is None and isinstance(value, torch.Tensor):
                    first_dtype = value.dtype
        keyword = dict(keyword)
        for i, arg in enumerate(schema.arguments):
            if i >= len(args) and arg.kwarg_only and "Tensor" in str(arg.type) \
                    and arg.name not in keyword:
                keyword[arg.name] = torch.empty(0, dtype=first_dtype or torch.float32)
        return positional, keyword
    except Exception:
        return list(args), {}


def meta_probe(op, args: list):
    """Run `op` on meta versions of `args` and return its output spec, or None if the
    call is invalid. This both validates a candidate node and yields the shape/dtype the
    next node will be pinned against.

    Meta only, never real tensors: shape inference with no data and no kernel keeps
    generation cheap, and keeps a broken kernel from taking the worker down. Ops the
    runtime has no meta rule for are handled by supplying one — see meta_impls.py.

    The shape is read from THIS overload's own return, never from a functional sibling's.
    A `.out` variant declared `-> ()` (split_copy.Tensor_out is the only one in the core
    tier) writes into its `Tensor[] out=` and returns None, so borrowing a shape from
    `split_copy.default` would register the node as a producer whose emitted call yields
    nothing — and every consumer wired to it then gets None at run time. An op with no
    return of its own has no output to offer, so the candidate is rejected here.
    """
    try:
        with torch.no_grad():
            positional, keyword = _split_by_schema(op.fn, [_to_meta(a) for a in args])
            keyword = {k: _to_meta(v) for k, v in keyword.items()}
            return _first_tensor(op.fn(*positional, **keyword))
    except Exception:
        return None
