"""lookup.py — an op name → the torch callable.

Deciding WHICH overload a mangled C++ symbol denotes does not happen here; that is resolved
offline and recorded in table_data.py (see table.py). All that is left is turning the name the
table already settled on into something callable.
"""

from __future__ import annotations

from typing import Any


def callable_for(op_name: str) -> Any | None:
    """torch.ops.aten.<base>.<overload> for an op name, or None if it doesn't resolve.

    A name with no overload suffix prefers the `default` overload over the packet: both
    the emitter and the schema-driven argument splitting need a `_schema`, which the
    packet does not carry.
    """
    import torch

    base, _, overload = op_name.partition(".")
    ns = getattr(torch.ops.aten, base, None)
    if ns is None:
        return None
    if overload:
        return getattr(ns, overload, None)
    return getattr(ns, "default", None) or ns
