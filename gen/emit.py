"""emit.py — helpers for rendering a graph as a graph-DEFINITION script.

These render the structural part only (op calls, leaves, arg literals). The
emitted file contains NO execution logic — running it is the execute/ concern
(execute/runner.py loads the file's `g` / `LEAVES` / `_BACKEND` and runs the diff).
"""

from __future__ import annotations

import math

import torch


def _qualname(op_name: str) -> str:
    """Source for the op's callable. 'add.Tensor' → torch.ops.aten.add.Tensor;
    'matmul' → torch.ops.aten.matmul.default; 'var.' (trailing dot = resolved to
    the OpOverloadPacket) → torch.ops.aten.var (the packet, which dispatches)."""
    if "." in op_name:
        base, ov = op_name.split(".", 1)
        if ov == "":  # trailing dot → packet (matches what find_op_for_symbol returns)
            return f"torch.ops.aten.{base}"
        return f"torch.ops.aten.{base}.{ov}"
    return f"torch.ops.aten.{op_name}.default"


def _const_repr(v) -> str | None:
    """Source text for a baked (non-tensor-port) argument, or None if not emittable."""
    if isinstance(v, float) and not math.isfinite(v):
        # repr(float('inf'))=='inf' etc. are bare names, not valid literals →
        # NameError when the emitted graph runs. Render them explicitly.
        return f"float({repr(repr(v))})"  # float('inf') / float('-inf') / float('nan')
    if v is None or isinstance(v, (bool, int, float, str)):
        return repr(v)
    if isinstance(v, (torch.dtype, torch.layout, torch.memory_format)):
        return str(v)  # 'torch.float32', 'torch.strided', ...
    if isinstance(v, torch.Tensor):  # e.g. a Tensor[] element (index= list), inline tensor
        # A concrete inline tensor must BAKE its values: `_make` is torch.randn/randint,
        # and inside g() torch.export traces that as an aten::rand* op which leaks into
        # the .pte (the runtime has no randint.low_out etc.). A baked tensor traces to a
        # constant instead. Meta out= buffers carry no values (their RNG is dead after
        # out= functionalization) → keep _make; empty tensors → a typed empty.
        if v.is_meta or v.numel() == 0:
            # An out= scratch buffer (meta, no values — overwritten by the op) or an
            # empty tensor: use torch.empty, NOT _make. _make is torch.randn/randint, and
            # an int out= buffer would otherwise leak a randint op into the .pte.
            return f"torch.empty({tuple(v.shape)}, dtype={v.dtype})"
        return f"torch.tensor({v.tolist()}, dtype={v.dtype})"
    if isinstance(v, (list, tuple)):
        parts = [_const_repr(x) for x in v]
        if any(p is None for p in parts):
            return None
        body = ", ".join(parts)
        return f"[{body}]" if isinstance(v, list) else f"({body},)"
    return None
