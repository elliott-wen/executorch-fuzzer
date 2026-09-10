"""render.py — a Graph, written out as a standalone Python script.

The corpus stores each graph as source defining `g` and `LEAVES` and nothing else: no
execution, no comparison. That is what lets one corpus entry be replayed by the eager
reference, by a device runner, or by hand — and read by a person when it fails.

Everything here is about producing text: naming an op's callable, writing a baked argument
back as a literal, and emitting the small prelude the script needs to rebuild its leaves.
"""

from __future__ import annotations

import math

import torch

# Per-float-leaf chance of injecting a boundary value (NaN / ±inf / ±0) into some of its
# elements. Leaves otherwise come from torch.randn and are always finite, so without this
# a single-op graph could never feed an op a non-finite INPUT — and a whole class of bugs
# is gated on exactly that (clamp/min/max/relu laundering NaN, exp(NaN), rsqrt(±0)). Kept
# low so it barely costs yield while every op still meets the boundary domain eventually.
NONFINITE_PROB = 0.0005


def qualname(op_name: str) -> str:
    """Source text for an op's callable.

    'add.Tensor' → torch.ops.aten.add.Tensor; 'matmul' → torch.ops.aten.matmul.default. A
    name with no overload gets `.default` rather than the bare packet, because the emitted
    call must name a single overload: the packet re-dispatches on the arguments, which can
    land somewhere other than what the constraints were solved against.
    """
    base, _, overload = op_name.partition(".")
    return f"torch.ops.aten.{base}.{overload or 'default'}"


def _num_repr(x: float) -> str:
    """A float as source text. repr(inf) is the bare name 'inf', which is a NameError when
    the emitted script runs, so non-finite values go through the float() constructor."""
    return repr(x) if math.isfinite(x) else f"float({repr(repr(x))})"


def const_repr(value) -> str | None:
    """Source text for a baked (non-port) argument, or None if it cannot be emitted —
    which makes the whole graph un-emittable, since the call would be incomplete."""
    if isinstance(value, bool):                 # before int/float: a bool is both
        return repr(value)
    if isinstance(value, float):
        return _num_repr(value)
    if isinstance(value, complex):
        # Built component-wise rather than via repr(): a complex with a non-finite part
        # reprs as '(inf+0j)', which is the same NameError trap as a bare inf.
        return f"complex({_num_repr(value.real)}, {_num_repr(value.imag)})"
    if value is None or isinstance(value, (int, str)):
        return repr(value)
    if isinstance(value, (torch.dtype, torch.layout, torch.memory_format)):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _tensor_repr(value)
    if isinstance(value, (list, tuple)):
        parts = [const_repr(v) for v in value]
        if any(p is None for p in parts):
            return None
        body = ", ".join(parts)
        return f"[{body}]" if isinstance(value, list) else f"({body},)"
    return None


def _tensor_repr(t: torch.Tensor) -> str:
    """An inline tensor argument, with its VALUES baked in.

    Baking matters: the leaf helper is torch.randn/randint, and torch.export traces those
    into an aten::rand* node that ends up in the .pte — which the runtime has no kernel
    for. A literal tensor traces to a constant instead. The exceptions are buffers with no
    meaningful values (a meta `out=` scratch, or an empty tensor), which only need the
    right shape and dtype.
    """
    if t.is_meta or t.numel() == 0:
        return f"torch.empty({tuple(t.shape)}, dtype={t.dtype})"
    return f"torch.tensor({t.tolist()}, dtype={t.dtype})"


_MAKE_LEAF = '''def _make(shape, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        if dtype.itemsize == 1:  # float8: no normal_ kernel, so draw wide and narrow
            return torch.randn(shape, dtype=torch.float32).to(dtype)
        return torch.randn(shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype)
    lo = -4 if dtype.is_signed else 0  # unsigned dtypes reject a negative low
    return torch.randint(lo, 5, shape, dtype=dtype)'''

_INJECT_NONFINITE = '''_NONFIN_P = {prob!r}  # chance a float leaf gets boundary values

def _inject(t):
    if not t.dtype.is_floating_point or t.numel() == 0:
        return t  # nan/inf are only meaningful for real floats
    if torch.rand(1).item() >= _NONFIN_P:
        return t
    choices = [float('nan'), float('inf'), float('-inf'), 0.0, -0.0]
    v = choices[int(torch.randint(len(choices), (1,)).item())]
    mask = torch.rand(t.shape) < 0.5
    mask.reshape(-1)[0] = True  # guarantee at least one injected element
    return torch.where(mask, torch.full_like(t, v), t)'''

_FIRST_TENSOR = '''def _first(r):
    if isinstance(r, torch.Tensor): return r
    if isinstance(r, (tuple, list)):
        for x in r:
            if isinstance(x, torch.Tensor): return x
    return r'''


def _leaf_lines(graph, inject: bool) -> list[str]:
    """The `L0 = _make(...)` lines that rebuild the graph's inputs."""
    lines = []
    for i, t in enumerate(graph.leaves):
        make = f"_make({tuple(t.shape)}, {t.dtype})"
        lines.append(f"L{i} = _inject({make})" if inject else f"L{i} = {make}")
    return lines


def _call_lines(graph) -> list[str] | None:
    """One `nX = ...` line per node, or None if an argument can't be rendered.

    Arguments are split into positional and keyword by the op's real JIT schema: anything
    after `*` (notably `out=`) MUST be passed by keyword or the call raises.
    """
    lines = []
    for nid, (op, slots) in enumerate(graph.nodes):
        exprs = []
        for kind, value in slots:
            if kind == "leaf":
                exprs.append(f"L{value}")
            elif kind == "ref":
                exprs.append(f"n{value}")
            else:
                rendered = const_repr(value)
                if rendered is None:
                    return None
                exprs.append(rendered)
        schema = getattr(op.fn, "_schema", None)
        if schema is None:
            return None
        positional, keyword = [], []
        for idx, _param in enumerate(op.params):
            if idx >= len(exprs) or idx >= len(schema.arguments):
                break
            arg = schema.arguments[idx]
            if getattr(arg, "kwarg_only", False):
                keyword.append(f"{arg.name}={exprs[idx]}")
            else:
                positional.append(exprs[idx])
        call = f"{qualname(op.op_name)}({', '.join(positional + keyword)})"
        lines.append(f"n{nid} = _first({call})")
    return lines


def render(graph, seed: int = 0, nonfinite_prob: float | None = None) -> str | None:
    """A Graph as a standalone script, or None if some baked argument cannot be written
    as source — which makes the graph unusable, since the call would be incomplete."""
    if nonfinite_prob is None:
        nonfinite_prob = NONFINITE_PROB
    calls = _call_lines(graph)
    if calls is None:
        return None
    inject = bool(nonfinite_prob and nonfinite_prob > 0.0)
    leaves = ", ".join(f"L{i}" for i in range(len(graph.leaves)))
    sinks = ", ".join("n%d" % s for s in graph.sink_ids())
    blocks = [
        f'"""Auto-generated — graph definition.\nGraph: {graph.describe()}\n"""',
        "import torch",
        f"torch.manual_seed({seed})  # reproducible leaf data",
        _MAKE_LEAF,
        _INJECT_NONFINITE.format(prob=float(nonfinite_prob)) if inject else "",
        _FIRST_TENSOR,
        "\n".join(_leaf_lines(graph, inject)),
        f"def g({leaves}):\n" + "\n".join(f"    {c}" for c in calls)
        + f"\n    return ({sinks},)",
        f"LEAVES = [{leaves}]",
    ]
    return "\n\n".join(b for b in blocks if b) + "\n"
